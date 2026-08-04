"""User-facing Turkish names for technical model features."""

from __future__ import annotations


FEATURE_LABELS = {
    "Administrative": "Hesap ve işlem adımı görüntüleme sayısı",
    "Administrative_Duration": "Hesap ve işlem adımlarında geçirilen süre (sn)",
    "Informational": "Yardım ve bilgi içeriği görüntüleme sayısı",
    "Informational_Duration": "Yardım ve bilgi içeriklerinde geçirilen süre (sn)",
    "ProductRelated": "Ürün inceleme sayfası görüntüleme sayısı",
    "ProductRelated_Duration": "Ürün inceleme sayfalarında geçirilen süre (sn)",
    "BounceRates": "Tek sayfadan ayrılma oranı",
    "ExitRates": "Sayfa sonrası siteden çıkış oranı",
    "PageValues": "Ziyaret edilen sayfaların dönüşüm değeri",
    "SpecialDay": "Kampanya veya özel güne yakınlık",
    "OperatingSystems": "İşletim sistemi grubu",
    "Browser": "Tarayıcı grubu",
    "Region": "Bölge grubu",
    "TrafficType": "Ziyaret kaynağı grubu",
    "VisitorType": "Ziyaretçi geçmişi",
    "Weekend": "Ziyaret hafta sonunda mı?",
    "Month": "Ziyaret ayı",
}

FEATURE_HELP = {
    "Administrative": "Hesap, giriş ve işlem süreci gibi ürün dışı operasyonel sayfalardan kaçının görüntülendiğini belirtir.",
    "Administrative_Duration": "Bu hesap ve işlem sayfalarında geçirilen toplam süredir.",
    "Informational": "Yardım, hakkımızda, iletişim veya politika gibi bilgilendirici içeriklerden kaçının görüntülendiğini belirtir.",
    "Informational_Duration": "Yardım ve bilgilendirme içeriklerinde geçirilen toplam süredir.",
    "ProductRelated": "Ürün listeleme, detay veya inceleme sayfalarındaki toplam görüntüleme sayısıdır.",
    "ProductRelated_Duration": "Ürünle ilgili sayfalarda geçirilen toplam süredir.",
    "BounceRates": "0 ile 0,20 arasında, ziyaretçinin başka bir sayfaya geçmeden ayrılma eğilimini gösteren analytics oranıdır.",
    "ExitRates": "Görüntülenen sayfaların oturumdaki son sayfa olma eğilimini gösteren analytics oranıdır.",
    "PageValues": "Analytics sistemi tarafından hesaplanan, işlem öncesi sayfaların dönüşüme katkı değeridir; kullanıcı tarafından tahmin edilmemelidir.",
    "SpecialDay": "0 özel günden uzak, 1 özel güne çok yakın anlamına gelir.",
    "OperatingSystems": "Veri setindeki anonim kategori kodudur; gerçek uygulamada analytics sisteminden otomatik gelmelidir.",
    "Browser": "Veri setindeki anonim tarayıcı kategori kodudur.",
    "Region": "Veri setindeki anonim coğrafi bölge kodudur.",
    "TrafficType": "Reklam, arama veya doğrudan ziyaret gibi kaynağın veri setindeki anonim kodudur.",
    "VisitorType": "Ziyaretçinin yeni, geri dönen veya diğer grupta olduğunu belirtir.",
    "Weekend": "Oturumun hafta sonuna denk gelip gelmediğini belirtir.",
    "Month": "Oturumun gerçekleştiği ayı belirtir.",
}

ENGINEERED_LABELS = {
    "TotalPages": "Toplam görüntülenen sayfa",
    "TotalDuration": "Sitede geçirilen toplam süre",
    "AdminDurationPerPage": "Hesap/işlem sayfası başına süre",
    "InfoDurationPerPage": "Yardım/bilgi sayfası başına süre",
    "ProductDurationPerPage": "Ürün sayfası başına süre",
    "DurationPerPage": "Sayfa başına ortalama süre",
    "ProductPageShare": "Ürün sayfalarının ziyaret içindeki payı",
    "AdminPageShare": "Hesap/işlem sayfalarının ziyaret içindeki payı",
    "InfoPageShare": "Yardım/bilgi sayfalarının ziyaret içindeki payı",
    "ExitBounceGap": "Çıkış ve tek sayfadan ayrılma farkı",
    "RetentionScore": "Sitede kalma skoru",
    "EngagementScore": "Etkileşim skoru",
    "ProductEngagement": "Ürün etkileşim skoru",
    "LogAdministrativeDuration": "Hesap/işlem süresinin ölçeklenmiş değeri",
    "LogInformationalDuration": "Yardım/bilgi süresinin ölçeklenmiş değeri",
    "LogProductDuration": "Ürün inceleme süresinin ölçeklenmiş değeri",
    "LogTotalDuration": "Toplam sürenin ölçeklenmiş değeri",
    "LogProductPages": "Ürün sayfası sayısının ölçeklenmiş değeri",
    "LogPageValues": "Dönüşüm değerinin ölçeklenmiş değeri",
    "MonthSin": "Yıl içindeki dönemsel ay etkisi (sinüs)",
    "MonthCos": "Yıl içindeki dönemsel ay etkisi (kosinüs)",
    "HasAdministrative": "Hesap/işlem sayfası ziyaret edildi mi?",
    "HasInformational": "Yardım/bilgi sayfası ziyaret edildi mi?",
    "HasPageValue": "Dönüşüm değeri oluştu mu?",
    "PageValuePerProductPage": "Ürün sayfası başına dönüşüm değeri",
}


def feature_label(name: str) -> str:
    """Translate raw/preprocessed/engineered names where a friendly label exists."""
    cleaned = str(name).split("__", 1)[-1]
    mapping = {**FEATURE_LABELS, **ENGINEERED_LABELS}
    if cleaned in mapping:
        return mapping[cleaned]
    category_prefixes = {
        "Month_": "Ziyaret ayı: ", "VisitorType_": "Ziyaretçi geçmişi: ",
        "TrafficType_": "Ziyaret kaynağı grubu: ", "Region_": "Bölge grubu: ",
        "Browser_": "Tarayıcı grubu: ", "OperatingSystems_": "İşletim sistemi grubu: ",
        "Weekend_": "Hafta sonu: ",
    }
    month_names = {"Feb": "Şubat", "Mar": "Mart", "May": "Mayıs", "June": "Haziran",
                   "Jul": "Temmuz", "Aug": "Ağustos", "Sep": "Eylül", "Oct": "Ekim",
                   "Nov": "Kasım", "Dec": "Aralık"}
    for prefix, label in category_prefixes.items():
        if cleaned.startswith(prefix):
            value = cleaned[len(prefix):]
            return label + month_names.get(value, value.replace("_", " "))
    for technical in sorted(mapping, key=len, reverse=True):
        if technical in cleaned:
            return cleaned.replace(technical, mapping[technical])
    return cleaned.replace("_", " ")
