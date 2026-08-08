"""
IndicTrans2 MT Architecture

Production-grade wrapper around the AI4Bharat IndicTrans2 models via HuggingFace.
Supports the 1B and 200M parameter variants with disjoint vocabularies for
handling extreme orthographic divergence across Indic scripts.

Key features:
- HuggingFace AutoModelForSeq2SeqLM backend (primary)
- IndicProcessor for Cython-optimized script unification
- Beam search generation with length penalty
- CTranslate2 backend support for edge inference (optional)
- Checkpoint management (save/load/from_pretrained)

Reference: architecture-documenattion.md (IndicTrans2 NMT)
Reference: detailed-report.md Phase 2 (MT subsystem design)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import yaml
from loguru import logger


# ─── FLORES-200 Language Codes ────────────────────────────────────────────────
# IndicTrans2 uses FLORES-200 codes: lang_Script
FLORES_LANG_CODES = {
    "hindi": "hin_Deva",
    "english": "eng_Latn",
    "marwari": "hin_Deva",  # Closest proxy — no dedicated FLORES code for dialects
    "mewari": "hin_Deva",
    "dhundhari": "hin_Deva",
    "hadoti": "hin_Deva",
    "mewati": "hin_Deva",
    "bagri": "hin_Deva",
    "gujarati": "guj_Gujr",
    "rajasthani": "hin_Deva",
    # Standard codes
    "hi": "hin_Deva",
    "en": "eng_Latn",
    "gu": "guj_Gujr",
}


def _get_flores_code(lang: str) -> str:
    """Map a language name or code to FLORES-200 format."""
    lang_lower = lang.lower().strip()
    if "_" in lang_lower and len(lang_lower) == 8:
        return lang_lower  # Already in FLORES format
    return FLORES_LANG_CODES.get(lang_lower, "hin_Deva")


class IndicTrans2MT(nn.Module):
    """
    Production wrapper for the IndicTrans2 Transformer architecture.

    Supports two modes:
    1. HuggingFace mode (default): Uses AutoModelForSeq2SeqLM + AutoTokenizer
    2. Skeleton mode (fallback): Uses a local Transformer when HuggingFace
       models are not available (e.g., offline development, testing)

    Usage:
        # Load pretrained IndicTrans2
        model = IndicTrans2MT(config_path="config/mt.yaml")

        # Translate
        translations = model.translate(
            ["नमस्ते, आप कैसे हैं?"],
            src_lang="hindi",
            tgt_lang="english",
        )

        # Fine-tune mode
        model.train()
        outputs = model(input_ids, attention_mask, labels=labels)
        loss = outputs.loss
    """

    # HuggingFace model identifiers for IndicTrans2 variants
    MODEL_REGISTRY = {
        "indic-indic-1B": "ai4bharat/indictrans2-indic-indic-1B",
        "indic-en-1B": "ai4bharat/indictrans2-indic-en-1B",
        "en-indic-1B": "ai4bharat/indictrans2-en-indic-1B",
        "indic-indic-dist-200M": "ai4bharat/indictrans2-indic-indic-dist-200M",
        "indic-en-dist-200M": "ai4bharat/indictrans2-indic-en-dist-200M",
        "en-indic-dist-200M": "ai4bharat/indictrans2-en-indic-dist-200M",
    }

    def __init__(
        self,
        config_path: str = "config/mt.yaml",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.config_path = Path(config_path)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.config = cfg.get("mt", {})
        else:
            logger.warning(
                f"MT config not found at {self.config_path}. Using defaults."
            )
            self.config = {}

        self.parameters_size = self.config.get("parameters", "1B")
        if device:
            self._device = device
        elif torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        # Model components (lazy-loaded)
        self._hf_model = None
        self._tokenizer = None
        self._processor = None
        self._model_name = model_name

        # Skeleton fallback (for offline dev/testing)
        self._skeleton_model = None

        # Generation config
        self._gen_config = {
            "num_beams": self.config.get("generation", {}).get("num_beams", 5),
            "max_length": self.config.get("generation", {}).get("max_length", 256),
            "length_penalty": self.config.get("generation", {}).get("length_penalty", 1.0),
            "early_stopping": True,
        }

        logger.info(
            f"IndicTrans2MT initialized | params={self.parameters_size} "
            f"device={self._device} vocab={self.config.get('vocab', 'disjoint')}"
        )

    # ─── Lazy Model Loading ───────────────────────────────────────────────────

    def _resolve_model_name(self) -> str:
        """Determine which HuggingFace model to load."""
        if self._model_name:
            # Check registry first (e.g. "indic-indic-dist-200M" → full HF ID)
            return self.MODEL_REGISTRY.get(self._model_name, self._model_name)
        # Default: Indic-to-Indic for dialect work
        variant = self.config.get("hf_model_name", "indic-indic-1B")
        return self.MODEL_REGISTRY.get(variant, variant)

    def _load_hf_model(self) -> bool:
        """Attempt to load the HuggingFace model. Returns True on success."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_name = self._resolve_model_name()
            logger.info(f"Loading IndicTrans2 from HuggingFace: {model_name}")

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
            )
            self._hf_model.to(self._device)

            # Try loading IndicProcessor for script unification
            try:
                from IndicTransToolkit import IndicProcessor

                self._processor = IndicProcessor(inference=True)
                logger.info("IndicProcessor loaded (Cython script unification)")
            except ImportError:
                logger.warning(
                    "IndicTransToolkit not installed. Script unification disabled. "
                    "Install with: pip install IndicTransToolkit"
                )
                self._processor = None

            logger.success(f"IndicTrans2 loaded: {model_name}")
            return True

        except Exception as e:
            logger.warning(
                f"Could not load primary model ({model_name}): {e}. "
                f"Attempting fallback to open non-gated translation model (facebook/m2m100_418M)..."
            )
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                fallback_name = "facebook/m2m100_418M"
                self._tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                    fallback_name,
                    torch_dtype=torch.float32,
                )
                self._hf_model.to(self._device)
                logger.success(f"Loaded open non-gated fallback model: {fallback_name}")
                return True
            except Exception as fallback_err:
                logger.warning(
                    f"Fallback model failed: {fallback_err}. "
                    f"Falling back to skeleton model for development."
                )
                return False

    def _ensure_model(self):
        """Ensure model is loaded (lazy initialization)."""
        if self._hf_model is not None:
            return

        if not self._load_hf_model():
            self._init_skeleton()

    def _init_skeleton(self):
        """Initialize skeleton model for offline development/testing."""
        logger.warning(
            "Using SKELETON model. Outputs will be meaningless. "
            "Install transformers + download IndicTrans2 for real inference."
        )
        d_model = 512
        self._skeleton_model = nn.ModuleDict({
            "encoder": nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
                num_layers=6,
            ),
            "decoder": nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=d_model, nhead=8, batch_first=True),
                num_layers=6,
            ),
        })

    @property
    def is_loaded(self) -> bool:
        """Whether the real HuggingFace model is loaded."""
        return self._hf_model is not None

    # ─── Forward Pass (Training) ──────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass for training.

        When the HuggingFace model is loaded, delegates directly to it.
        Returns a Seq2SeqLMOutput with .loss and .logits attributes.
        """
        self._ensure_model()

        if self._hf_model is not None:
            return self._hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        else:
            # Skeleton: return a dummy loss for development
            dummy_loss = torch.tensor(0.0, requires_grad=True, device=self._device)
            return type("Output", (), {"loss": dummy_loss, "logits": None})()

    # ─── Translation (Inference) ──────────────────────────────────────────────

    @torch.inference_mode()
    def translate(
        self,
        texts: list[str],
        src_lang: str,
        tgt_lang: str,
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> list[str]:
        """
        Translate a batch of texts from src_lang to tgt_lang.

        This is the primary inference API. It handles:
        1. IndicProcessor preprocessing (script unification)
        2. Tokenization with source language tags
        3. Beam search generation
        4. Target-side detokenization
        5. IndicProcessor postprocessing

        Args:
            texts: List of source language texts
            src_lang: Source language (e.g., "hindi", "marwari", "en")
            tgt_lang: Target language (e.g., "english", "hindi", "en")
            num_beams: Override beam width (default from config)
            max_length: Override max generation length (default from config)

        Returns:
            List of translated texts
        """
        self._ensure_model()

        if self._hf_model is None:
            logger.warning("Skeleton model active. Returning placeholder translations.")
            return [f"[SKELETON] Translation of: {t[:50]}..." for t in texts]

        src_flores = _get_flores_code(src_lang)
        tgt_flores = _get_flores_code(tgt_lang)

        # Step 1: IndicProcessor preprocessing
        if self._processor is not None:
            processed_texts = self._processor.preprocess_batch(
                texts, src_lang=src_flores, tgt_lang=tgt_flores
            )
        else:
            processed_texts = texts

        # Step 2: Tokenize
        inputs = self._tokenizer(
            processed_texts,
            padding="longest",
            truncation=True,
            max_length=self._gen_config["max_length"],
            return_tensors="pt",
        ).to(self._device)

        # Step 3: Generate
        generated_ids = self._hf_model.generate(
            **inputs,
            num_beams=num_beams or self._gen_config["num_beams"],
            max_length=max_length or self._gen_config["max_length"],
            length_penalty=self._gen_config["length_penalty"],
            early_stopping=self._gen_config["early_stopping"],
        )

        # Step 4: Decode
        translations = self._tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        # Step 5: IndicProcessor postprocessing
        if self._processor is not None:
            translations = self._processor.postprocess_batch(
                translations, lang=tgt_flores
            )

        return translations

    # ─── Model Souping ────────────────────────────────────────────────────────

    def create_model_soup(
        self,
        checkpoint_paths: list[str | Path],
        base_weight: float = 0.5,
    ) -> IndicTrans2MT:
        """
        Average the weights of multiple fine-tuned checkpoints with the base model.

        Model souping (Wortsman et al., 2022) prevents catastrophic forgetting
        by blending the fine-tuned dialect-specific weights back toward the
        general-domain base weights.

        Args:
            checkpoint_paths: Paths to fine-tuned checkpoint state_dicts
            base_weight: Weight given to the base model (0.5 = equal blend)

        Returns:
            A new IndicTrans2MT with the souped weights
        """
        self._ensure_model()

        if self._hf_model is None:
            logger.warning("Model souping requires loaded HF model. Returning self.")
            return self

        logger.info(
            f"Creating model soup from {len(checkpoint_paths)} checkpoints "
            f"(base_weight={base_weight})"
        )

        base_state = copy.deepcopy(self._hf_model.state_dict())

        # Load and average fine-tuned checkpoints
        ft_weight = (1.0 - base_weight) / max(len(checkpoint_paths), 1)
        souped_state = {}

        for key in base_state:
            souped_state[key] = base_state[key].float() * base_weight

        for ckpt_path in checkpoint_paths:
            ckpt_state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            for key in souped_state:
                if key in ckpt_state:
                    souped_state[key] += ckpt_state[key].float() * ft_weight

        # Cast back to original dtype
        for key in souped_state:
            souped_state[key] = souped_state[key].to(base_state[key].dtype)

        # Create new model with souped weights
        souped_model = IndicTrans2MT(
            config_path=str(self.config_path),
            model_name=self._resolve_model_name(),
            device=self._device,
        )
        souped_model._ensure_model()
        if souped_model._hf_model is not None:
            souped_model._hf_model.load_state_dict(souped_state)

        logger.success(
            f"Model soup created: {len(checkpoint_paths)} checkpoints "
            f"+ base (weight={base_weight})"
        )
        return souped_model

    # ─── Checkpoint Management ────────────────────────────────────────────────

    def save_checkpoint(self, path: str | Path) -> Path:
        """Save model weights and tokenizer to disk."""
        self._ensure_model()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self._hf_model is not None:
            self._hf_model.save_pretrained(path)
            self._tokenizer.save_pretrained(path)
            logger.info(f"Model + tokenizer saved to {path}")
        else:
            torch.save(self._skeleton_model.state_dict(), path / "skeleton.pt")
            logger.info(f"Skeleton weights saved to {path}")
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        """Load model weights from a checkpoint directory."""
        path = Path(path)
        if (path / "config.json").exists():
            # HuggingFace checkpoint
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(path), trust_remote_code=True
            )
            self._hf_model = AutoModelForSeq2SeqLM.from_pretrained(
                str(path), trust_remote_code=True
            )
            self._hf_model.to(self._device)
            logger.info(f"Loaded HF checkpoint from {path}")
        elif (path / "skeleton.pt").exists():
            self._init_skeleton()
            self._skeleton_model.load_state_dict(
                torch.load(path / "skeleton.pt", map_location="cpu")
            )
            logger.info(f"Loaded skeleton checkpoint from {path}")
        else:
            raise FileNotFoundError(f"No valid checkpoint found in {path}")

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        self._ensure_model()
        model = self._hf_model or self._skeleton_model
        if model is None:
            return 0
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def freeze_encoder(self) -> None:
        """Freeze encoder weights for dialect fine-tuning (Months 5-6 roadmap)."""
        self._ensure_model()
        if self._hf_model is not None and hasattr(self._hf_model, "model"):
            for param in self._hf_model.model.encoder.parameters():
                param.requires_grad = False
            frozen = sum(1 for p in self._hf_model.model.encoder.parameters())
            logger.info(f"Froze {frozen} encoder parameter groups")

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        self._ensure_model()
        model = self._hf_model or self._skeleton_model
        if model is not None:
            for param in model.parameters():
                param.requires_grad = True
            logger.info("All parameters unfrozen")
