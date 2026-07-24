import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, LogitsProcessorList
from subtitle_nmt_thesis.models.baseline import CPLLogitsProcessor

FLORES_CODES = {
    "en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng",
    "ta": "tam_Taml", "te": "tel_Telu", "mr": "mar_Deva",
    "gu": "guj_Gujr", "kn": "kan_Knda", "ml": "mal_Mlym",
    "pa": "pan_Guru", "or": "ory_Orya", "ur": "urd_Arab",
    "as": "asm_Beng", "mai": "mai_Deva", "sat": "sat_Olck",
    "ks": "kas_Arab", "sd": "snd_Arab", "ne": "npi_Deva",
    "kok": "gom_Deva", "doi": "doi_Deva", "mni": "mni_Mtei",
    "brx": "brx_Deva", "san": "san_Deva",
}


def resolve_flores(code: str) -> str:
    if "_" in code:
        return code
    return FLORES_CODES.get(code, code)


class NLLBTranslator:
    def __init__(
        self,
        model_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "eng_Latn",
        tgt_lang: str = "hin_Deva",
        device: str = "cpu",
    ):
        self.src_lang = resolve_flores(src_lang)
        self.tgt_lang = resolve_flores(tgt_lang)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.eos_id = self.tokenizer.eos_token_id or 2

    def _generate(self, texts: list[str], num_beams: int = 5, num_return_sequences: int = 1, logits_processor=None) -> list[str]:
        self.tokenizer.src_lang = self.src_lang
        inputs = self.tokenizer(texts, padding="longest", truncation=True, max_length=256, return_tensors="pt").to(self.device)
        forced_bos = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        gen_kwargs = dict(
            forced_bos_token_id=forced_bos,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            max_length=256,
        )
        if logits_processor is not None:
            gen_kwargs["logits_processor"] = logits_processor
        outputs = self.model.generate(**inputs, **gen_kwargs)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    def translate(self, text: str, source_lang: str = "", target_lang: str = "") -> str:
        return self._generate([text], num_beams=5)[0]

    def translate_n(self, text: str, n: int = 5, source_lang: str = "", target_lang: str = "") -> list[str]:
        return self._generate([text], num_beams=n, num_return_sequences=n)

    def translate_constrained(self, text: str, max_cpl: int = 42) -> str:
        processor = CPLLogitsProcessor(max_cpl=max_cpl, eos_token_id=self.eos_id)
        return self._generate([text], logits_processor=LogitsProcessorList([processor]))[0]
