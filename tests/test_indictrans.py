from subtitle_nmt_thesis.models.indictrans import resolve_flores, IndicTransTranslator


def test_resolve_flores():
    assert resolve_flores("hi") == "hin_Deva"
    assert resolve_flores("en") == "eng_Latn"
    assert resolve_flores("eng_Latn") == "eng_Latn"
    assert resolve_flores("ta") == "tam_Taml"
    assert resolve_flores("te") == "tel_Telu"
