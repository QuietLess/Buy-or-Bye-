"""Interactive Buy or Bye purchase-intent demo."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain import explain_single_prediction
from src.dashboard import calculate_what_if, score_batch
from src.data import FEATURE_COLUMNS
from src.predict import load_artifacts, predict_session
from src.segments import build_segments
from src.ui_labels import FEATURE_HELP, FEATURE_LABELS, feature_label

st.set_page_config(page_title="Buy or Bye", page_icon="🛒", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1180px; padding-top: 2rem;}
  [data-testid="stMetric"] {background:#f8fafc; border:1px solid #e2e8f0; padding:1rem; border-radius:14px;}
  .hero {padding:1.3rem 1.5rem; border-radius:18px; color:white; background:linear-gradient(120deg,#0f172a,#0f766e); margin-bottom:1.25rem;}
  .hero h1 {margin:0; font-size:2.2rem;} .hero p {margin:.5rem 0 0;color:#ccfbf1;}
</style>
<div class="hero"><h1>🛒 Buy or Bye</h1><p>E-ticaret oturumlarını dönüşüm olasılığına göre önceliklendirin.</p></div>
""", unsafe_allow_html=True)


@st.cache_resource
def artifacts():
    return load_artifacts()


try:
    model, metadata = artifacts()
except FileNotFoundError as exc:
    st.error(str(exc)); st.code("python -m src.train"); st.stop()

with st.sidebar:
    st.header("Model bilgisi")
    st.write(f"**Model:** {metadata['model_name']}")
    st.write(f"**Karar eşiği:** {metadata['threshold']:.2f}")
    if "test_metrics" in metadata:
        st.metric("Test PR-AUC", f"{metadata['test_metrics']['pr_auc']:.3f}")
        st.metric("Test ROC-AUC", f"{metadata['test_metrics']['roc_auc']:.3f}")
    with st.expander("Segment ve aksiyon kuralları"):
        for segment in build_segments(metadata["threshold"]):
            upper = min(segment.upper, 1.0)
            st.markdown(
                f"**{segment.label}** · `{segment.lower:.2f}–{upper:.2f}`  \n"
                f"{segment.action}"
            )
    st.caption("Tahminler karar desteğidir; kullanıcıya otomatik olumsuz işlem uygulamak için kullanılmamalıdır.")

DEFAULT_INPUT = {
    "Administrative": 0, "Administrative_Duration": 0.0,
    "Informational": 0, "Informational_Duration": 0.0,
    "ProductRelated": 12, "ProductRelated_Duration": 480.0,
    "BounceRates": 0.002899, "ExitRates": 0.025000, "PageValues": 0.50,
    "SpecialDay": 0.0, "OperatingSystems": 1, "Browser": 1,
    "Region": 1, "TrafficType": 1, "VisitorType": "Returning_Visitor",
    "Weekend": False, "Month": "Nov",
}
ANALYTICS_DEMO_VALUES = {
    "BounceRates": 0.002899,
    "ExitRates": 0.025000,
    "PageValues": 0.50,
    "SpecialDay": 0.0,
}
WHAT_IF_RANGES = {
    "Administrative": (0.0, 50.0), "Administrative_Duration": (0.0, 5000.0),
    "Informational": (0.0, 50.0), "Informational_Duration": (0.0, 5000.0),
    "ProductRelated": (0.0, 500.0), "ProductRelated_Duration": (0.0, 50000.0),
}


def save_and_show(fig, filename: str) -> None:
    """Persist a dashboard figure headlessly, render it, then release memory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / filename, dpi=160, bbox_inches="tight")
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


manual_tab, batch_tab, insights_tab, guide_tab = st.tabs([
    "✍️ Manuel tahmin & What-if", "📄 CSV performans testi",
    "🧠 Modeli ne etkiliyor?", "ℹ️ Kullanım rehberi",
])

with manual_tab:
    st.subheader("Oturum özelliklerini girin")
    with st.form("session_form"):
        tab_pages, tab_analytics, tab_context = st.tabs([
            "Sayfa etkileşimi", "Analitik metrikleri", "Ziyaretçi bağlamı",
        ])
        with tab_pages:
            c1, c2, c3 = st.columns(3)
            administrative = c1.number_input(FEATURE_LABELS["Administrative"], 0, 50, 0, help=FEATURE_HELP["Administrative"])
            admin_duration = c1.number_input(FEATURE_LABELS["Administrative_Duration"], 0.0, 5000.0, 0.0, help=FEATURE_HELP["Administrative_Duration"])
            informational = c2.number_input(FEATURE_LABELS["Informational"], 0, 50, 0, help=FEATURE_HELP["Informational"])
            info_duration = c2.number_input(FEATURE_LABELS["Informational_Duration"], 0.0, 5000.0, 0.0, help=FEATURE_HELP["Informational_Duration"])
            product_related = c3.number_input(FEATURE_LABELS["ProductRelated"], 0, 500, 12, help=FEATURE_HELP["ProductRelated"])
            product_duration = c3.number_input(FEATURE_LABELS["ProductRelated_Duration"], 0.0, 50000.0, 480.0, help=FEATURE_HELP["ProductRelated_Duration"])
        with tab_analytics:
            bounce_rates = ANALYTICS_DEMO_VALUES["BounceRates"]
            exit_rates = ANALYTICS_DEMO_VALUES["ExitRates"]
            page_values = ANALYTICS_DEMO_VALUES["PageValues"]
            special_day = ANALYTICS_DEMO_VALUES["SpecialDay"]
            st.info(
                "Bu alanlar kullanıcının doğrudan bileceği bilgiler değildir; gerçek sistemde web analytics "
                "tarafından otomatik sağlanır. Demo sonuçlarının karşılaştırılabilir olması için sabit tutulur."
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tek sayfadan ayrılma", f"{bounce_rates:.4f}")
            c2.metric("Siteden çıkış", f"{exit_rates:.4f}")
            c3.metric("Dönüşüm değeri", f"{page_values:.2f}")
            c4.metric("Özel güne yakınlık", f"{special_day:.1f}")
            st.caption(
                "Ayrılma ve çıkış değerleri veri medyanlarıdır. Page Value=0.50, diğer davranış "
                "alanlarının etkisini gözlemlemek için seçilmiş kontrollü düşük demo referansıdır; "
                "gerçek ölçüm veya veri seti ortalaması değildir."
            )
            with st.expander("Page Value nedir?"):
                st.write(
                    "Page Value, analytics sisteminin bir sayfanın satın alma gibi bir dönüşüme ne kadar "
                    "katkı sağladığını geçmiş oturumlardan tahmin eden sayısal değerdir. Kullanıcının elle "
                    "girdiği bir puan değildir ve ürün fiyatı anlamına gelmez."
                )
        with tab_context:
            c1, c2, c3 = st.columns(3)
            visitor_names = {"Returning_Visitor": "Daha önce ziyaret etmiş", "New_Visitor": "İlk kez ziyaret ediyor", "Other": "Diğer/bilinmiyor"}
            visitor_type = c1.selectbox(FEATURE_LABELS["VisitorType"], list(visitor_names), format_func=visitor_names.get, help=FEATURE_HELP["VisitorType"])
            weekend = c1.toggle(FEATURE_LABELS["Weekend"], help=FEATURE_HELP["Weekend"])
            month_names = {"Feb": "Şubat", "Mar": "Mart", "May": "Mayıs", "June": "Haziran", "Jul": "Temmuz", "Aug": "Ağustos", "Sep": "Eylül", "Oct": "Ekim", "Nov": "Kasım", "Dec": "Aralık"}
            month = c2.selectbox(FEATURE_LABELS["Month"], list(month_names), index=8, format_func=month_names.get, help=FEATURE_HELP["Month"])
            operating_system = c2.selectbox(FEATURE_LABELS["OperatingSystems"], range(1, 9), help=FEATURE_HELP["OperatingSystems"])
            browser = c3.selectbox(FEATURE_LABELS["Browser"], range(1, 14), help=FEATURE_HELP["Browser"])
            region = c3.selectbox(FEATURE_LABELS["Region"], range(1, 10), help=FEATURE_HELP["Region"])
            traffic_type = c3.selectbox(FEATURE_LABELS["TrafficType"], range(1, 21), help=FEATURE_HELP["TrafficType"])
            st.caption("Grup kodları anonim veri setinden gelir; gerçek sistemde analytics tarafından otomatik doldurulmalıdır.")
        submitted = st.form_submit_button("Oturumu analiz et", type="primary", use_container_width=True)

    if submitted:
        st.session_state["manual_input"] = {
            "Administrative": administrative, "Administrative_Duration": admin_duration,
            "Informational": informational, "Informational_Duration": info_duration,
            "ProductRelated": product_related, "ProductRelated_Duration": product_duration,
            "BounceRates": bounce_rates, "ExitRates": exit_rates, "PageValues": page_values,
            "SpecialDay": special_day, "OperatingSystems": operating_system, "Browser": browser,
            "Region": region, "TrafficType": traffic_type, "VisitorType": visitor_type,
            "Weekend": weekend, "Month": month,
        }

    if "manual_input" in st.session_state:
        input_data = pd.DataFrame([st.session_state["manual_input"]])
        result = predict_session(input_data, model, metadata)
        c1, c2, c3 = st.columns(3)
        c1.metric("Satın alma olasılığı", f"%{result['probability'] * 100:.1f}")
        c2.metric("Niyet segmenti", result["segment_label"])
        c3.metric("Binary sinyal", "Satın alma" if result["purchase_prediction"] else "Satın almama")
        st.info(f"**Önerilen aksiyon:** {result['business_action']}")
        st.caption(f"İş hedefi: {result['business_objective']}")
        st.caption("Bu sonuç, sabit analytics demo değerleri ile girdiğiniz davranış alanlarının birleşimidir.")

        st.divider()
        left, right = st.columns([1.15, .85])
        with left:
            st.subheader("What-if senaryosu")
            feature = st.selectbox("Değiştirilecek davranış", list(WHAT_IF_RANGES), format_func=feature_label)
            st.caption(FEATURE_HELP.get(feature, ""))
            default_lower, default_upper = WHAT_IF_RANGES[feature]
            r1, r2 = st.columns(2)
            lower = r1.number_input("Senaryo alt sınırı", value=default_lower, key=f"whatif_min_{feature}")
            upper = r2.number_input("Senaryo üst sınırı", value=default_upper, key=f"whatif_max_{feature}")
            try:
                scenario = calculate_what_if(input_data, model, feature, lower, upper)
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot(scenario[feature], scenario["purchase_probability"], color="#0f766e", lw=2)
                for segment in build_segments(metadata["threshold"]):
                    ax.axhspan(segment.lower, min(segment.upper, 1), color=segment.color, alpha=.07)
                current = float(input_data.iloc[0][feature])
                ax.scatter([current], [result["probability"]], color="#7c3aed", zorder=3, label="Mevcut oturum")
                ax.axhline(metadata["threshold"], color="#f97316", ls="--", label="Yüksek niyet sınırı")
                ax.set(xlabel=feature_label(feature), ylabel="Satın alma olasılığı", ylim=(0, 1))
                ax.grid(alpha=.2); ax.legend()
                save_and_show(fig, "what_if.png")
                st.caption("Diğer tüm alanlar sabit tutulur. Grafik ilişkiyi gösterir; nedensellik kanıtlamaz.")
            except ValueError as exc:
                st.error(str(exc))
        with right:
            st.subheader("Tahmini yönlendiren etkenler")
            try:
                if hasattr(model, "lightgbm_weight"):
                    st.warning(
                        f"Final model bir ensemble'dır. Bu SHAP grafiği yalnız Engineered LightGBM "
                        f"bileşenini (ensemble ağırlığı %{model.lightgbm_weight * 100:.0f}) açıklar; "
                        "ensemble tahmininin tamamını açıklamaz."
                    )
                factors = explain_single_prediction(model, input_data)
                factor_frame = pd.DataFrame(factors, columns=["Özellik", "Etki"])
                factor_frame["Özellik"] = factor_frame["Özellik"].map(feature_label)
                factor_frame["Yön"] = np.where(factor_frame["Etki"] >= 0, "Satın almaya doğru", "Satın almadan uzağa")
                local_plot = factor_frame.iloc[:8].sort_values("Etki")
                fig, ax = plt.subplots(figsize=(6.5, 5))
                colors = np.where(local_plot["Etki"] >= 0, "#10b981", "#ef4444")
                ax.barh(local_plot["Özellik"], local_plot["Etki"], color=colors)
                ax.axvline(0, color="#475569", lw=1)
                ax.set(xlabel="SHAP katkısı (ham model skoru)", title="Bu oturumun lokal açıklaması")
                ax.grid(axis="x", alpha=.2)
                save_and_show(fig, "local_shap.png")
                st.caption("Yeşil satın alma skorunu yükselten, kırmızı düşüren faktördür. Çubuk büyüklüğü olasılık yüzdesi değildir.")
                with st.expander("SHAP değerlerini tablo olarak gör"):
                    st.dataframe(factor_frame, hide_index=True, use_container_width=True)
                st.caption("SHAP model katkısını açıklar; nedensellik veya önerilen müdahalenin sonucunu göstermez.")
            except Exception as exc:
                st.warning(f"Yerel açıklama üretilemedi: {exc}")

with batch_tab:
    st.subheader("Kendi CSV verinizi yükleyin")
    st.write("17 özellik zorunludur. `Revenue` isteğe bağlıdır; varsa performans, yoksa yalnız tahmin üretilir.")
    template = pd.DataFrame([DEFAULT_INPUT]).loc[:, FEATURE_COLUMNS]
    st.download_button("CSV şablonunu indir", template.to_csv(index=False).encode("utf-8"),
                       "buy_or_bye_template.csv", "text/csv")
    with st.expander("CSV sütunlarının Türkçe açıklamaları"):
        dictionary = pd.DataFrame({
            "CSV sütunu": FEATURE_COLUMNS,
            "Kullanıcı dostu adı": [FEATURE_LABELS[name] for name in FEATURE_COLUMNS],
            "Açıklama": [FEATURE_HELP[name] for name in FEATURE_COLUMNS],
        })
        st.dataframe(dictionary, hide_index=True, use_container_width=True)
    uploaded = st.file_uploader("CSV dosyası", type=["csv"])
    if uploaded is not None:
        try:
            uploaded_data = pd.read_csv(uploaded)
            scored, metrics = score_batch(uploaded_data, model, metadata)
            st.success(f"{len(scored):,} satır başarıyla skorlandı.")
            st.dataframe(scored.head(100), hide_index=True, use_container_width=True)
            st.download_button("Skorlanmış CSV'yi indir", scored.to_csv(index=False).encode("utf-8"),
                               "buy_or_bye_scored.csv", "text/csv", type="primary")

            if metrics is None:
                st.info("Revenue sütunu bulunmadığı için performans metriği hesaplanmadı.")
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.hist(scored["purchase_probability"], bins=25, color="#0f766e", edgecolor="white")
                ax.set(title="Tahmin Olasılığı Dağılımı", xlabel="Satın alma olasılığı", ylabel="Oturum")
                ax.grid(axis="y", alpha=.2)
            else:
                st.subheader("Yüklenen veri performansı")
                metric_columns = st.columns(6)
                values = [metrics["roc_auc"], metrics["pr_auc"], metrics["precision"],
                          metrics["recall"], metrics["f1"], metrics["accuracy"]]
                names = ["ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]
                for column, name, value in zip(metric_columns, names, values):
                    column.metric(name, "N/A" if value is None else f"{value:.3f}")
                if metrics["roc_auc"] is None:
                    st.warning("Revenue tek sınıf içeriyor; ROC-AUC ve PR-AUC tanımsızdır.")
                fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
                matrix = np.asarray(metrics["confusion_matrix"])
                image = axes[0].imshow(matrix, cmap="Blues")
                for i in range(2):
                    for j in range(2):
                        axes[0].text(j, i, matrix[i, j], ha="center", va="center")
                axes[0].set(title=f"Confusion Matrix (eşik={metadata['threshold']:.2f})",
                            xlabel="Tahmin", ylabel="Gerçek", xticks=[0, 1], yticks=[0, 1],
                            xticklabels=["Bye", "Buy"], yticklabels=["Bye", "Buy"])
                fig.colorbar(image, ax=axes[0], fraction=.046)
                axes[1].hist(scored["purchase_probability"], bins=25, color="#0f766e", edgecolor="white")
                axes[1].set(title="Tahmin Olasılığı Dağılımı", xlabel="Satın alma olasılığı", ylabel="Oturum")
                axes[1].grid(axis="y", alpha=.2)
            save_and_show(fig, "batch_evaluation.png")
        except Exception as exc:
            st.error(f"CSV işlenemedi: {exc}")

with insights_tab:
    st.subheader("Model genelinde en etkili özellikler")
    report_path = PROJECT_ROOT / "reports" / "global_shap_importance.json"
    if not report_path.exists():
        st.warning("Global SHAP raporu henüz oluşturulmamış.")
        st.code("python -m src.shap_report")
    else:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = pd.DataFrame(report["features"])
        if report.get("trained_at") != metadata.get("trained_at"):
            st.warning("SHAP raporu mevcut model sürümüyle eşleşmiyor; raporu yeniden üretin.")
        if report.get("explanation_scope") == "engineered_lightgbm_component":
            st.warning(
                f"Global SHAP yalnız Engineered LightGBM bileşenini "
                f"(%{report.get('explained_component_weight', 0) * 100:.0f} ensemble ağırlığı) açıklar; "
                "Random Forest katkısını kapsamaz."
            )
        top = rows.head(12).sort_values("mean_abs_shap")
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(top["display_name"], top["mean_abs_shap"], color="#7c3aed")
        ax.set(xlabel="Ortalama mutlak SHAP", title="Modelin global özellik önemi")
        ax.grid(axis="x", alpha=.2)
        save_and_show(fig, "global_shap_dashboard.png")
        st.info("Bu grafik modelin tüm örneklerde hangi bilgilere daha fazla dayandığını gösterir. Büyük değer, daha güçlü genel etki demektir; etkinin her kullanıcıda aynı yönde olduğu anlamına gelmez.")
        display = rows.head(15).loc[:, ["display_name", "mean_abs_shap"]].rename(columns={
            "display_name": "Özellik", "mean_abs_shap": "Ortalama |SHAP|",
        })
        st.dataframe(display, hide_index=True, use_container_width=True)
        st.warning("Page Value ailesi çok etkili olduğundan gerçek zamanlı kullanımda veri sızıntısı ve erişilebilirlik ayrıca kontrol edilmelidir.")

with guide_tab:
    st.subheader("Hangi test neyi ölçer?")
    st.markdown("""
- **Manuel tahmin:** Tek oturumun olasılığını, segmentini ve aksiyonunu gösterir. Model performansı ölçmez.
- **What-if:** Seçtiğiniz tek özelliği değiştirir, diğerlerini sabit tutar. Duyarlılık analizidir; nedensel etki değildir.
- **Etiketsiz CSV:** Çok sayıda oturumu skorlar ve segmentler; doğruluk metriği üretmez.
- **Etiketli CSV:** `Revenue` gerçek sonucuyla ROC-AUC, PR-AUC, precision, recall, F1, accuracy ve confusion matrix üretir.
- **Lokal SHAP:** Belirli bir oturumun skorunu hangi faktörlerin yukarı veya aşağı ittiğini gösterir.
- **Global SHAP:** Modelin genel olarak hangi özelliklere en çok dayandığını gösterir.
    """)
    st.warning("Gerçek performans değerlendirmesi için eğitim verisinden bağımsız, mümkünse daha yeni tarihli veri kullanın.")
