from src.ui_labels import feature_label


def test_base_and_preprocessed_feature_names_are_humanised():
    assert feature_label("Administrative") == "Hesap ve işlem adımı görüntüleme sayısı"
    assert feature_label("numeric__ProductRelated_Duration") == "Ürün inceleme sayfalarında geçirilen süre (sn)"


def test_engineered_feature_name_is_humanised():
    assert feature_label("numeric__ProductPageShare") == "Ürün sayfalarının ziyaret içindeki payı"
