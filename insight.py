"""
Data Science Workflow Studio - 実務特化型データ分析アプリ
直感的なステップバイステップで問題定義から解釈まで実行

必要なパッケージ:
pip install streamlit pandas numpy scipy scikit-learn plotly statsmodels

実行:
streamlit run insight.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="Insight flow Studio",
    layout="wide",
    initial_sidebar_state='expanded'
)

# スタイル
st.markdown("""
<style>
    [data-testid='stAppViewContainer'] {
        background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #ffffff;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #8b5cf6, #ec4899);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .step-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #8b5cf6;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #8b5cf6;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #064e3b;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10b981;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# セッションステート初期化
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'column_mapping' not in st.session_state:
    st.session_state.column_mapping = {}
if 'selected_target' not in st.session_state:
    st.session_state.selected_target = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

# ユーティリティ関数
@st.cache_data
def load_csv(file_buf):
    """CSVファイルを読み込む"""
    try:
        file_buf.seek(0)
        df = pd.read_csv(file_buf)
        return df
    except Exception as e:
        st.error(f"ファイル読み込みエラー: {e}")
        return None

def calculate_vif(df, features):
    """VIF（多重共線性）を計算"""
    try:
        X = df[features].select_dtypes(include=[np.number]).dropna()
        if X.shape[1] < 2:
            return None
        vif_data = pd.DataFrame()
        vif_data["変数"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data.sort_values('VIF', ascending=False)
    except:
        return None

# メインアプリ
st.markdown('<div class="main-header">📊 Data Science Workflow Studio</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">問題定義から解釈まで、実務に特化したデータサイエンスワークフロー</p>', unsafe_allow_html=True)

# サイドバー: ステップナビゲーション
with st.sidebar:
    st.header("📋 分析ステップ")
    steps = [
        "1️⃣ 問題定義",
        "2️⃣ データ理解と点検",
        "3️⃣ データ前処理・EDA",
        "4️⃣ モデリングと推定",
        "5️⃣ 解釈とレポート"
    ]
    
    current_step = st.radio("ステップを選択", steps, index=st.session_state.step - 1)
    st.session_state.step = steps.index(current_step) + 1
    
    st.markdown("---")
    st.markdown("### 💡 現在の状態")
    if st.session_state.df is not None:
        st.success(f"✅ データ読み込み済 ({st.session_state.df.shape[0]}行)")
    if st.session_state.selected_target:
        st.success(f"✅ 目的変数: {st.session_state.selected_target}")
    if st.session_state.selected_features:
        st.success(f"✅ 説明変数: {len(st.session_state.selected_features)}個")

# ==================== ステップ1: 問題定義 ====================
if st.session_state.step == 1:
    st.markdown('<div class="step-header">1️⃣ 問題定義: 何のための分析ですか？</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 分析を始める前に考えるべきこと</h4>
    <ul>
        <li><strong>ビジネス課題</strong>: 解決したい実務上の問題は何ですか？</li>
        <li><strong>分析目的</strong>: 記述、推論、予測、因果推定のどれを目指しますか？</li>
        <li><strong>期待される成果</strong>: この分析で何がわかれば成功ですか？</li>
        <li><strong>意思決定への影響</strong>: 結果をどう活用しますか？</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 分析の目的を記述してください")
        problem_statement = st.text_area(
            "ビジネス課題と分析目的",
            height=200,
            placeholder="例: 売上を向上させるため、どの要因が売上に最も影響を与えているかを特定したい。",
            key="problem_statement"
        )
        
        analysis_goal = st.selectbox(
            "分析の主な目的",
            [
                "選択してください",
                "📊 記述統計（現状把握・要約）",
                "🔍 推論（仮説検定・関係性の検証）",
                "🎯 予測（将来の値を予測）",
                "⚡ 因果推定（施策効果の測定）",
                "🔬 探索的分析（パターン発見）"
            ],
            key="analysis_goal"
        )
        
        expected_outcome = st.text_area(
            "期待される成果・アクション",
            height=150,
            placeholder="例: 重要な説明変数トップ3を特定し、それらに集中投資する戦略を立案する。",
            key="expected_outcome"
        )
    
    with col2:
        st.subheader("📌 分析設計のヒント")
        st.markdown("""
        <div style="background-color: #1e293b; padding: 1rem; border-radius: 10px;">
        <h5>記述統計の場合</h5>
        <p>→ 平均、中央値、分布の可視化</p>
        
        <h5>推論の場合</h5>
        <p>→ t検定、ANOVA、回帰分析で関係性を検証</p>
        
        <h5>予測の場合</h5>
        <p>→ 機械学習モデルで精度重視</p>
        
        <h5>因果推定の場合</h5>
        <p>→ DID、IV、RCTで施策効果を測定</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if problem_statement and analysis_goal != "選択してください":
        st.markdown('<div class="success-box">✅ 問題定義が完了しました！次のステップに進んでください。</div>', unsafe_allow_html=True)
        if st.button("➡️ ステップ2へ進む", type="primary"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.warning("⚠️ 分析目的を明確にしてから次のステップに進んでください。")

# ==================== ステップ2: データ理解と点検 ====================
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">2️⃣ データ理解と点検</div>', unsafe_allow_html=True)
    
    # ファイルアップロード
    uploaded_file = st.file_uploader("📁 CSVファイルをアップロード", type=['csv'])
    
    # サンプルデータ生成ボタン
    col_sample1, col_sample2 = st.columns([1, 3])
    with col_sample1:
        if st.button('サンプルデータ生成'):
            sample_df = pd.DataFrame({
                'age': np.random.randint(22, 65, 500),
                'income': (np.random.normal(50000, 12000, 500)).astype(int),
                'experience': np.random.randint(0, 30, 500),
                'education': np.random.choice(['HS', 'BSc', 'MSc', 'PhD'], 500),
                'group': np.random.choice(['A', 'B', 'C'], 500),
                'outcome': np.random.normal(100, 20, 500)
            })
            st.session_state.df = sample_df
            st.success("✅ サンプルデータを生成しました")
            st.rerun()
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success(f"✅ データ読み込み完了: {df.shape[0]}行 × {df.shape[1]}列")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # データ概要
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総行数", df.shape[0])
        col2.metric("総列数", df.shape[1])
        col3.metric("数値列", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("欠損値あり", df.isnull().any().sum())
        
        st.markdown("---")
        
        # タブで機能を分割
        tab1, tab2, tab3, tab4 = st.tabs(["📋 データプレビュー", "🏷️ 変数名編集", "🎯 変数選択", "📊 基本統計と可視化"])
        
        with tab1:
            st.subheader("データプレビュー")
            st.dataframe(df.head(100), use_container_width=True)
            
            st.subheader("データ型情報")
            dtype_df = pd.DataFrame({
                '列名': df.columns,
                'データ型': df.dtypes.values,
                '欠損数': df.isnull().sum().values,
                '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2).values,
                'ユニーク数': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with tab2:
            st.subheader("🏷️ 変数名をわかりやすく編集")
            st.info("💡 分析しやすいように変数名を日本語や意味のある名前に変更できます")
            
            if not st.session_state.column_mapping:
                st.session_state.column_mapping = {col: col for col in df.columns}
            
            col_edit1, col_edit2 = st.columns(2)
            
            with col_edit1:
                st.markdown("**元の列名**")
                for col in df.columns:
                    st.text(col)
            
            with col_edit2:
                st.markdown("**新しい列名**")
                for col in df.columns:
                    new_name = st.text_input(
                        f"rename_{col}",
                        value=st.session_state.column_mapping.get(col, col),
                        label_visibility="collapsed",
                        key=f"rename_{col}"
                    )
                    st.session_state.column_mapping[col] = new_name
            
            if st.button("✅ 変数名を確定", type="primary"):
                # 重複チェック
                new_names = list(st.session_state.column_mapping.values())
                if len(new_names) != len(set(new_names)):
                    st.error("❌ エラー: 重複する変数名があります")
                else:
                    df_renamed = df.rename(columns=st.session_state.column_mapping)
                    st.session_state.df = df_renamed
                    st.success("✅ 変数名を更新しました！")
                    st.rerun()
        
        with tab3:
            st.subheader("🎯 目的変数と説明変数の選択")
            st.markdown("""
            <div class="info-box">
            <strong>重回帰分析のために</strong>、分析に使用する変数を選択してください。
            <ul>
                <li><strong>目的変数（従属変数）</strong>: 予測・説明したい変数</li>
                <li><strong>説明変数（独立変数）</strong>: 目的変数に影響を与える変数</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # 目的変数選択
            target_col = st.selectbox(
                "目的変数を選択",
                options=['選択してください'] + list(df.columns),
                key="target_selection"
            )
            
            if target_col != '選択してください':
                st.session_state.selected_target = target_col
                
                # 説明変数候補（目的変数以外）
                feature_candidates = [col for col in df.columns if col != target_col]
                
                st.markdown("---")
                st.markdown("#### 説明変数を選択")
                
                # 全選択・全解除ボタン
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ すべて選択"):
                        st.session_state.selected_features = feature_candidates
                        st.rerun()
                with col_btn2:
                    if st.button("❌ すべて解除"):
                        st.session_state.selected_features = []
                        st.rerun()
                
                # 説明変数選択
                selected_features = st.multiselect(
                    "使用する説明変数",
                    options=feature_candidates,
                    default=[f for f in st.session_state.selected_features if f in feature_candidates] if st.session_state.selected_features else feature_candidates,
                    key="feature_selection"
                )
                st.session_state.selected_features = selected_features
                
                if selected_features:
                    st.success(f"✅ 目的変数: **{target_col}** | 説明変数: **{len(selected_features)}個**")
                    
                    # 選択した変数のサマリ
                    st.markdown("#### 選択した変数のサマリ")
                    selected_df = df[[target_col] + selected_features]
                    st.dataframe(selected_df.describe(), use_container_width=True)
        
        with tab4:
            st.subheader("📊 基本統計量と可視化")
            
            if st.session_state.selected_features:
                viz_col = st.selectbox(
                    "可視化する変数を選択",
                    options=[st.session_state.selected_target] + st.session_state.selected_features
                )
                
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    if pd.api.types.is_numeric_dtype(df[viz_col]):
                        fig = px.histogram(
                            df, x=viz_col,
                            title=f"分布: {viz_col}",
                            marginal="box"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        vc = df[viz_col].value_counts().reset_index()
                        vc.columns = [viz_col, 'count']
                        fig = px.bar(vc, x=viz_col, y='count', title=f"頻度: {viz_col}")
                        st.plotly_chart(fig, use_container_width=True)
                
                with col_viz2:
                    st.markdown("**基本統計量**")
                    if pd.api.types.is_numeric_dtype(df[viz_col]):
                        stats_dict = {
                            '平均': df[viz_col].mean(),
                            '中央値': df[viz_col].median(),
                            '標準偏差': df[viz_col].std(),
                            '最小値': df[viz_col].min(),
                            '最大値': df[viz_col].max(),
                            '欠損数': df[viz_col].isnull().sum()
                        }
                        st.table(pd.DataFrame(stats_dict.items(), columns=['統計量', '値']))
                    else:
                        st.write(df[viz_col].value_counts())
                
                # 相関行列
                if len(st.session_state.selected_features) >= 2:
                    st.markdown("---")
                    st.markdown("#### 相関行列")
                    numeric_features = [col for col in st.session_state.selected_features if pd.api.types.is_numeric_dtype(df[col])]
                    if numeric_features:
                        corr_df = df[[st.session_state.selected_target] + numeric_features].corr()
                        fig_corr = px.imshow(
                            corr_df,
                            text_auto='.2f',
                            title="相関行列",
                            color_continuous_scale='RdBu_r'
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
        
        # 次のステップへ
        st.markdown("---")
        if st.session_state.selected_target and st.session_state.selected_features:
            if st.button("➡️ ステップ3へ進む（データ前処理）", type="primary"):
                st.session_state.step = 3
                st.rerun()
        else:
            st.warning("⚠️ 目的変数と説明変数を選択してください")
    else:
        st.info("💡 CSVファイルをアップロードするか、サンプルデータを生成してください")

# ==================== ステップ3: データ前処理・EDA ====================
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">3️⃣ データ前処理・実務特化EDA</div>', unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ ステップ2でデータをアップロードしてください")
        if st.button("⬅️ ステップ2に戻る"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    
    df = st.session_state.df.copy()
    target = st.session_state.selected_target
    features = st.session_state.selected_features
    
    if not target or not features:
        st.warning("⚠️ ステップ2で目的変数と説明変数を選択してください")
        if st.button("⬅️ ステップ2に戻る"):
            st.session_state.step = 2
            st.rerun()
        st.stop()
    
    # 作業用データフレーム
    if st.session_state.processed_df is None:
        st.session_state.processed_df = df[[target] + features].copy()
    
    work_df = st.session_state.processed_df.copy()
    
    st.info(f"🎯 目的変数: **{target}** | 説明変数: **{len(features)}個**")
    
    # 前処理タブ
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 欠損値処理",
        "📉 外れ値処理",
        "🔄 データ変換",
        "🏷️ カテゴリ変数処理",
        "💾 処理済みデータ"
    ])
    
    with tab1:
        st.subheader("🔧 欠損値補完")
        
        # 欠損値の確認
        missing_info = pd.DataFrame({
            '変数': work_df.columns,
            '欠損数': work_df.isnull().sum().values,
            '欠損率(%)': (work_df.isnull().sum() / len(work_df) * 100).round(2).values
        })
        missing_info = missing_info[missing_info['欠損数'] > 0]
        
        if len(missing_info) > 0:
            st.dataframe(missing_info, use_container_width=True)
            
            st.markdown("---")
            impute_method = st.selectbox(
                "補完方法を選択",
                ["選択してください", "平均値", "中央値", "最頻値", "削除"]
            )
            
            if impute_method != "選択してください":
                impute_cols = st.multiselect(
                    "補完する変数を選択",
                    options=missing_info['変数'].tolist()
                )
                
                if st.button("✅ 補完実行", type="primary") and impute_cols:
                    for col in impute_cols:
                        if impute_method == "平均値":
                            work_df[col].fillna(work_df[col].mean(), inplace=True)
                        elif impute_method == "中央値":
                            work_df[col].fillna(work_df[col].median(), inplace=True)
                        elif impute_method == "最頻値":
                            work_df[col].fillna(work_df[col].mode()[0], inplace=True)
                        elif impute_method == "削除":
                            work_df.dropna(subset=[col], inplace=True)
                    
                    st.session_state.processed_df = work_df
                    st.success(f"✅ {len(impute_cols)}個の変数の欠損値を補完しました")
                    st.rerun()
        else:
            st.success("✅ 欠損値はありません")
    
    with tab2:
        st.subheader("📉 外れ値処理")
        
        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            outlier_col = st.selectbox("外れ値を確認する変数", options=numeric_cols)
            
            # 外れ値の可視化
            fig = px.box(work_df, y=outlier_col, title=f"{outlier_col} の箱ひげ図")
            st.plotly_chart(fig, use_container_width=True)
            
            # IQR法で外れ値検出
            Q1 = work_df[outlier_col].quantile(0.25)
            Q3 = work_df[outlier_col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = work_df[(work_df[outlier_col] < lower_bound) | (work_df[outlier_col] > upper_bound)]
            
            st.write(f"**外れ値検出**: {len(outliers)}件 ({len(outliers)/len(work_df)*100:.2f}%)")
            st.write(f"下限: {lower_bound:.2f}, 上限: {upper_bound:.2f}")
            
            st.markdown("---")
            outlier_method = st.selectbox(
                "外れ値処理方法",
                ["選択してください", "Winsorization（境界値で置換）", "Trimming（削除）", "対数変換"]
            )
            
            if outlier_method != "選択してください":
                if st.button("✅ 外れ値処理実行", type="primary"):
                    if outlier_method == "Winsorization（境界値で置換）":
                        work_df[outlier_col] = work_df[outlier_col].clip(lower=lower_bound, upper=upper_bound)
                        st.success("✅ Winsorization完了")
                    elif outlier_method == "Trimming（削除）":
                        work_df = work_df[(work_df[outlier_col] >= lower_bound) & (work_df[outlier_col] <= upper_bound)]
                        st.success(f"✅ {len(outliers)}件の外れ値を削除しました")
                    elif outlier_method == "対数変換":
                        work_df[outlier_col] = np.log1p(work_df[outlier_col].clip(lower=0))
                        st.success("✅ 対数変換完了")
                    
                    st.session_state.processed_df = work_df
                    st.rerun()
        else:
            st.info("数値変数がありません")
    
    with tab3:
        st.subheader("🔄 データ変換")
        
        numeric_cols = work_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            transform_cols = st.multiselect("変換する変数を選択", options=numeric_cols)
            
            if transform_cols:
                transform_method = st.selectbox(
                    "変換方法",
                    ["選択してください", "標準化（Z-score）", "正規化（Min-Max）", "対数変換", "平方根変換"]
                )
                
                if transform_method != "選択してください" and st.button("✅ 変換実行", type="primary"):
                    for col in transform_cols:
                        if transform_method == "標準化（Z-score）":
                            work_df[f"{col}_std"] = (work_df[col] - work_df[col].mean()) / work_df[col].std()
                        elif transform_method == "正規化（Min-Max）":
                            work_df[f"{col}_norm"] = (work_df[col] - work_df[col].min()) / (work_df[col].max() - work_df[col].min())
                        elif transform_method == "対数変換":
                            work_df[f"{col}_log"] = np.log1p(work_df[col].clip(lower=0))
                        elif transform_method == "平方根変換":
                            work_df[f"{col}_sqrt"] = np.sqrt(work_df[col].clip(lower=0))
                    
                    st.session_state.processed_df = work_df
                    st.success(f"✅ {len(transform_cols)}個の変数を変換しました")
                    st.rerun()
        else:
            st.info("数値変数がありません")
    
    with tab4:
        st.subheader("🏷️ カテゴリ変数のエンコーディング")
        
        cat_cols = work_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            st.write(f"カテゴリ変数: {len(cat_cols)}個")
            
            encode_col = st.selectbox("エンコードする変数", options=cat_cols)
            
            if encode_col:
                st.write(f"**{encode_col}** のユニーク値数: {work_df[encode_col].nunique()}")
                st.write("値の例:", work_df[encode_col].value_counts().head(10))
                
                encode_method = st.selectbox(
                    "エンコード方法",
                    ["選択してください", "Label Encoding（序数）", "One-Hot Encoding（ダミー変数）", "頻度エンコーディング"]
                )
                
                if encode_method != "選択してください" and st.button("✅ エンコード実行", type="primary"):
                    if encode_method == "Label Encoding（序数）":
                        le = LabelEncoder()
                        work_df[f"{encode_col}_encoded"] = le.fit_transform(work_df[encode_col].astype(str))
                        st.success(f"✅ {encode_col}をLabel Encodingしました")
                    elif encode_method == "One-Hot Encoding（ダミー変数）":
                        dummies = pd.get_dummies(work_df[encode_col], prefix=encode_col, drop_first=True)
                        work_df = pd.concat([work_df, dummies], axis=1)
                        st.success(f"✅ {encode_col}をOne-Hot Encodingしました（{len(dummies.columns)}個の新変数）")
                    elif encode_method == "頻度エンコーディング":
                        freq_map = work_df[encode_col].value_counts(normalize=True).to_dict()
                        work_df[f"{encode_col}_freq"] = work_df[encode_col].map(freq_map)
                        st.success(f"✅ {encode_col}を頻度エンコーディングしました")
                    
                    st.session_state.processed_df = work_df
                    st.rerun()
        else:
            st.info("カテゴリ変数がありません")
    
    with tab5:
        st.subheader("💾 処理済みデータ")
        
        st.write(f"**現在のデータ**: {work_df.shape[0]}行 × {work_df.shape[1]}列")
        
        # データプレビュー
        st.dataframe(work_df.head(100), use_container_width=True)
        
        # 統計サマリ
        st.markdown("---")
        st.markdown("#### 基本統計量")
        st.dataframe(work_df.describe(), use_container_width=True)
        
        # ダウンロード
        st.markdown("---")
        csv = work_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 処理済みデータをダウンロード (CSV)",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        # 処理を確定
        if st.button("✅ この処理済みデータで分析を続ける", type="primary"):
            st.session_state.processed_df = work_df
            st.success("✅ 処理済みデータを確定しました")
    
    # 次のステップへ
    st.markdown("---")
    if st.button("➡️ ステップ4へ進む（モデリング）", type="primary"):
        st.session_state.processed_df = work_df
        st.session_state.step = 4
        st.rerun()

# ==================== ステップ4: モデリングと推定 ====================
elif st.session_state.step == 4:
    st.markdown('<div class="step-header">4️⃣ モデリングと推定</div>', unsafe_allow_html=True)
    
    if st.session_state.processed_df is None:
        st.warning("⚠️ ステップ3でデータ前処理を完了してください")
        if st.button("⬅️ ステップ3に戻る"):
            st.session_state.step = 3
            st.rerun()
        st.stop()
    
    df = st.session_state.processed_df.copy()
    target = st.session_state.selected_target
    
    # 目的変数が存在するか確認
    if target not in df.columns:
        st.error(f"❌ 目的変数 '{target}' がデータに存在しません")
        st.stop()
    
    # 数値列のみを特徴量として使用
    numeric_features = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_features:
        st.error("❌ 数値型の説明変数がありません。ステップ3でカテゴリ変数をエンコードしてください")
        if st.button("⬅️ ステップ3に戻る"):
            st.session_state.step = 3
            st.rerun()
        st.stop()
    
    st.info(f"🎯 目的変数: **{target}** | 使用可能な説明変数: **{len(numeric_features)}個**")
    
    # 分析タイプ選択
    st.markdown("---")
    st.subheader("📊 分析目的を選択")
    
    analysis_type = st.selectbox(
        "実行したい分析",
        [
            "選択してください",
            "📊 記述統計（平均・分散・分布比較）",
            "🔍 推論（t検定・ANOVA・相関分析）",
            "📈 回帰分析（線形回帰・重回帰）",
            "🎯 予測モデル（機械学習）"
        ]
    )
    
    if analysis_type == "選択してください":
        st.warning("⚠️ 分析タイプを選択してください")
        st.stop()
    
    # 記述統計
    if analysis_type == "📊 記述統計（平均・分散・分布比較）":
        st.markdown("### 記述統計分析")
        
        desc_var = st.selectbox("分析する変数", options=[target] + numeric_features)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # ヒストグラム
            fig_hist = px.histogram(df, x=desc_var, title=f"{desc_var} の分布", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # 統計量
            stats_dict = {
                '平均': df[desc_var].mean(),
                '中央値': df[desc_var].median(),
                '標準偏差': df[desc_var].std(),
                '分散': df[desc_var].var(),
                '最小値': df[desc_var].min(),
                '最大値': df[desc_var].max(),
                '歪度': df[desc_var].skew(),
                '尖度': df[desc_var].kurtosis()
            }
            st.table(pd.DataFrame(stats_dict.items(), columns=['統計量', '値']))
    
    # 推論
    elif analysis_type == "🔍 推論（t検定・ANOVA・相関分析）":
        st.markdown("### 統計的推論")
        
        test_type = st.selectbox("検定タイプ", ["相関分析", "t検定（2群比較）"])
        
        if test_type == "相関分析":
            st.subheader("相関分析")
            
            corr_features = st.multiselect(
                "相関を分析する変数",
                options=[target] + numeric_features,
                default=[target] + numeric_features[:min(5, len(numeric_features))]
            )
            
            if len(corr_features) >= 2:
                corr_matrix = df[corr_features].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    title="相関行列",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # 最も相関が高いペア
                st.markdown("#### 相関が高い変数ペア（上位5組）")
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_pairs.append({
                            '変数1': corr_matrix.columns[i],
                            '変数2': corr_matrix.columns[j],
                            '相関係数': corr_matrix.iloc[i, j]
                        })
                corr_pairs_df = pd.DataFrame(corr_pairs).sort_values('相関係数', ascending=False, key=abs)
                st.dataframe(corr_pairs_df.head(5), use_container_width=True)
    
    # 回帰分析
    elif analysis_type == "📈 回帰分析（線形回帰・重回帰）":
        st.markdown("### 重回帰分析")
        
        st.info("💡 重回帰分析で説明変数の影響を定量化します")
        
        # 説明変数選択
        selected_features = st.multiselect(
            "回帰分析に使用する説明変数",
            options=numeric_features,
            default=numeric_features[:min(10, len(numeric_features))]
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ 説明変数を選択してください")
            st.stop()
        
        # 欠損値処理
        regression_df = df[[target] + selected_features].dropna()
        
        if len(regression_df) == 0:
            st.error("❌ 欠損値を除外するとデータが残りません")
            st.stop()
        
        st.write(f"**使用データ**: {len(regression_df)}行")
        
        # モデル選択
        model_type = st.selectbox(
            "モデルタイプ",
            ["通常の線形回帰（OLS）", "Ridge回帰（正則化）", "Lasso回帰（変数選択）"]
        )
        
        if st.button("✅ 回帰分析実行", type="primary"):
            X = regression_df[selected_features]
            y = regression_df[target]
            
            # 学習データとテストデータに分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # モデル構築
            if model_type == "通常の線形回帰（OLS）":
                # statsmodelsで詳細な結果を取得
                X_train_sm = sm.add_constant(X_train)
                model_sm = sm.OLS(y_train, X_train_sm).fit()
                
                st.markdown("#### 📊 回帰分析結果（視覚化）")
                
                # モデル適合度の表示
                st.markdown("##### 🎯 モデル全体の性能")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("R²", f"{model_sm.rsquared:.4f}")
                col_m2.metric("調整済みR²", f"{model_sm.rsquared_adj:.4f}")
                col_m3.metric("F統計量", f"{model_sm.fvalue:.2f}")
                col_m4.metric("p値(F)", f"{model_sm.f_pvalue:.4f}")
                
                # 係数の表とグラフ
                st.markdown("---")
                st.markdown("##### 📈 回帰係数の詳細")
                
                # 係数テーブル作成
                coef_df = pd.DataFrame({
                    '変数': ['切片'] + selected_features,
                    '係数': model_sm.params.values,
                    '標準誤差': model_sm.bse.values,
                    't値': model_sm.tvalues.values,
                    'p値': model_sm.pvalues.values,
                    '95%CI下限': model_sm.conf_int()[0].values,
                    '95%CI上限': model_sm.conf_int()[1].values
                })
                
                # 有意性の判定
                coef_df['有意性'] = coef_df['p値'].apply(
                    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'n.s.'
                )
                
                # 係数の絶対値で降順ソート（切片を除く）
                coef_df_sorted = pd.concat([
                    coef_df[coef_df['変数'] == '切片'],
                    coef_df[coef_df['変数'] != '切片'].sort_values('係数', key=abs, ascending=False)
                ])
                
                # 係数表示
                st.dataframe(
                    coef_df_sorted.style.format({
                        '係数': '{:.4f}',
                        '標準誤差': '{:.4f}',
                        't値': '{:.3f}',
                        'p値': '{:.4f}',
                        '95%CI下限': '{:.4f}',
                        '95%CI上限': '{:.4f}'
                    }),
                    use_container_width=True
                )
                
                st.caption("*** p<0.001, ** p<0.01, * p<0.05, n.s. 有意でない")
                
                # 係数の可視化（切片を除く）
                st.markdown("---")
                st.markdown("##### 📊 係数の可視化（信頼区間付き）")
                
                coef_plot_df = coef_df_sorted[coef_df_sorted['変数'] != '切片'].copy()
                
                fig_coef = go.Figure()
                
                # 係数のバー
                colors = ['red' if p < 0.05 else 'gray' for p in coef_plot_df['p値']]
                fig_coef.add_trace(go.Bar(
                    y=coef_plot_df['変数'],
                    x=coef_plot_df['係数'],
                    orientation='h',
                    marker=dict(color=colors),
                    error_x=dict(
                        type='data',
                        symmetric=False,
                        array=coef_plot_df['95%CI上限'] - coef_plot_df['係数'],
                        arrayminus=coef_plot_df['係数'] - coef_plot_df['95%CI下限']
                    ),
                    name='係数'
                ))
                
                fig_coef.add_vline(x=0, line_dash="dash", line_color="black", line_width=1)
                fig_coef.update_layout(
                    title="回帰係数と95%信頼区間（赤=有意、灰=非有意）",
                    xaxis_title="係数の値",
                    yaxis_title="変数",
                    height=max(300, len(coef_plot_df) * 50),
                    showlegend=False
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                
                # 解釈のヘルプ
                st.markdown("---")
                st.markdown("##### 💡 結果の解釈")
                st.info("""
                **係数の読み方:**
                - **正の係数**: その変数が1単位増えると、目的変数が係数分だけ増加
                - **負の係数**: その変数が1単位増えると、目的変数が係数分だけ減少
                - **p値 < 0.05**: 統計的に有意（偶然ではない可能性が高い）
                - **95%信頼区間**: 真の係数が存在する範囲（95%の確率）
                """)
                
                # 有意な変数のサマリ
                significant_vars = coef_df_sorted[(coef_df_sorted['p値'] < 0.05) & (coef_df_sorted['変数'] != '切片')]
                if len(significant_vars) > 0:
                    st.success(f"✅ **有意な変数（p < 0.05）**: {len(significant_vars)}個")
                    for idx, row in significant_vars.iterrows():
                        direction = "増加" if row['係数'] > 0 else "減少"
                        st.write(f"- **{row['変数']}**: 1単位増加すると目的変数が{abs(row['係数']):.4f}だけ{direction} (p={row['p値']:.4f})")
                else:
                    st.warning("⚠️ 統計的に有意な変数はありませんでした")
                
                # モデル診断
                st.markdown("---")
                st.markdown("##### 🔍 モデル診断")
                
                col_diag1, col_diag2 = st.columns(2)
                with col_diag1:
                    st.write("**Durbin-Watson統計量**:", f"{sm.stats.stattools.durbin_watson(model_sm.resid):.3f}")
                    st.caption("2に近いほど良好（自己相関なし）")
                
                with col_diag2:
                    if model_sm.condition_number > 30:
                        st.warning(f"⚠️ **条件数**: {model_sm.condition_number:.0f} (多重共線性の可能性)")
                    else:
                        st.success(f"✅ **条件数**: {model_sm.condition_number:.0f}")
                
                # 詳細な統計情報（折りたたみ）
                with st.expander("📋 詳細な統計情報（テキスト形式）"):
                    st.text(model_sm.summary())
                
                # 予測
                X_test_sm = sm.add_constant(X_test)
                y_pred = model_sm.predict(X_test_sm)
                
            elif model_type == "Ridge回帰（正則化）":
                alpha = st.slider("正則化パラメータ（α）", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 係数表示
                coef_df = pd.DataFrame({
                    '変数': selected_features,
                    '係数': model.coef_
                }).sort_values('係数', ascending=False, key=abs)
                st.dataframe(coef_df, use_container_width=True)
                
            else:  # Lasso
                alpha = st.slider("正則化パラメータ（α）", 0.01, 10.0, 1.0)
                model = Lasso(alpha=alpha)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 係数表示（ゼロでない変数のみ）
                coef_df = pd.DataFrame({
                    '変数': selected_features,
                    '係数': model.coef_
                })
                coef_df = coef_df[coef_df['係数'] != 0].sort_values('係数', ascending=False, key=abs)
                st.markdown("#### 選択された変数（係数≠0）")
                st.dataframe(coef_df, use_container_width=True)
            
            # 評価指標
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1, col2 = st.columns(2)
            col1.metric("R² スコア", f"{r2:.4f}")
            col2.metric("RMSE", f"{rmse:.4f}")
            
            # 予測vs実測プロット
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                name='予測値',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            fig_pred.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='理想線',
                line=dict(color='red', dash='dash')
            ))
            fig_pred.update_layout(
                title="実測値 vs 予測値",
                xaxis_title="実測値",
                yaxis_title="予測値"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # 残差プロット
            residuals = y_test - y_pred
            fig_resid = px.scatter(x=y_pred, y=residuals, title="残差プロット")
            fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
            fig_resid.update_xaxes(title="予測値")
            fig_resid.update_yaxes(title="残差")
            st.plotly_chart(fig_resid, use_container_width=True)
            
            # VIF（多重共線性チェック）
            st.markdown("---")
            st.markdown("#### 多重共線性チェック（VIF）")
            vif_data = calculate_vif(regression_df, selected_features)
            if vif_data is not None:
                st.dataframe(vif_data, use_container_width=True)
                st.caption("VIF > 10: 多重共線性の疑いあり")
            
            # 結果を保存
            st.session_state.model_results['regression'] = {
                'model_type': model_type,
                'r2': r2,
                'rmse': rmse,
                'features': selected_features,
                'predictions': y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred),
                'actual': y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
            }
    
    # 予測モデル
    elif analysis_type == "🎯 予測モデル（機械学習）":
        st.markdown("### 機械学習による予測")
        
        # モデル選択（高度な機能追加）
        model_category = st.selectbox(
            "モデルカテゴリ",
            ["基本モデル", "高度なモデル（XGBoost/LightGBM）"]
        )
        
        # 説明変数選択
        selected_features = st.multiselect(
            "予測に使用する説明変数",
            options=numeric_features,
            default=numeric_features[:min(10, len(numeric_features))]
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ 説明変数を選択してください")
            st.stop()
        
        # 欠損値処理
        ml_df = df[[target] + selected_features].dropna()
        
        # 目的変数の型判定
        is_classification = len(ml_df[target].unique()) < 20 and not pd.api.types.is_float_dtype(ml_df[target])
        
        task_type = "分類" if is_classification else "回帰"
        st.info(f"🎯 タスクタイプ: **{task_type}** （目的変数のユニーク数: {ml_df[target].nunique()}）")
        
        # 不均衡データ対策（分類の場合）
        if is_classification:
            class_distribution = ml_df[target].value_counts()
            imbalance_ratio = class_distribution.max() / class_distribution.min()
            
            if imbalance_ratio > 3:
                st.warning(f"⚠️ 不均衡データ検出: 最大クラス/最小クラス = {imbalance_ratio:.1f}倍")
                use_smote = st.checkbox("SMOTE（オーバーサンプリング）を使用", value=True)
            else:
                use_smote = False
        else:
            use_smote = False
        
        # 高度なオプション
        with st.expander("🔧 高度な設定"):
            test_size = st.slider("テストデータの割合", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("ランダムシード", value=42, min_value=0)
            
            if model_category == "高度なモデル（XGBoost/LightGBM）":
                enable_tuning = st.checkbox("ハイパーパラメータチューニングを実行", value=False)
                enable_shap = st.checkbox("SHAP分析を実行", value=True)
                enable_calibration = st.checkbox("確率キャリブレーションを実行", value=False)
            else:
                enable_tuning = False
                enable_shap = False
                enable_calibration = False
        
        # モデル選択
        if model_category == "基本モデル":
            if is_classification:
                ml_model = st.selectbox("モデル選択", ["ランダムフォレスト分類", "ロジスティック回帰"])
            else:
                ml_model = st.selectbox("モデル選択", ["ランダムフォレスト回帰", "線形回帰"])
        else:
            if is_classification:
                ml_model = st.selectbox("モデル選択", ["XGBoost分類", "LightGBM分類"])
            else:
                ml_model = st.selectbox("モデル選択", ["XGBoost回帰", "LightGBM回帰"])
        
        if st.button("✅ モデル学習実行", type="primary"):
            X = ml_df[selected_features]
            y = ml_df[target]
            
            # 学習データとテストデータに分割
            if is_classification:
                # 層化分割
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                for train_idx, test_idx in sss.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            with st.spinner("モデル学習中..."):
                # SMOTE適用
                if use_smote:
                    try:
                        from imblearn.over_sampling import SMOTE
                        smote = SMOTE(random_state=random_state)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.info(f"✅ SMOTE適用後の訓練データサイズ: {len(X_train)}行")
                    except Exception as e:
                        st.warning(f"SMOTE適用に失敗しました: {e}")
                
                # モデル構築
                if model_category == "基本モデル":
                    if is_classification:
                        if ml_model == "ランダムフォレスト分類":
                            model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                        else:
                            model = LogisticRegression(max_iter=1000)
                    else:
                        if ml_model == "ランダムフォレスト回帰":
                            model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                        else:
                            model = LinearRegression()
                    
                    model.fit(X_train, y_train)
                    
                else:  # 高度なモデル
                    try:
                        from xgboost import XGBClassifier, XGBRegressor
                        from lightgbm import LGBMClassifier, LGBMRegressor
                    except ImportError:
                        st.error("❌ XGBoostまたはLightGBMがインストールされていません")
                        st.code("pip install xgboost lightgbm")
                        st.stop()
                    
                    if is_classification:
                        if ml_model == "XGBoost分類":
                            base_model = XGBClassifier(random_state=random_state, eval_metric='logloss')
                        else:
                            base_model = LGBMClassifier(random_state=random_state, verbose=-1)
                    else:
                        if ml_model == "XGBoost回帰":
                            base_model = XGBRegressor(random_state=random_state)
                        else:
                            base_model = LGBMRegressor(random_state=random_state, verbose=-1)
                    
                    # ハイパーパラメータチューニング
                    if enable_tuning:
                        st.info("🔄 ハイパーパラメータチューニング実行中...")
                        
                        if 'XGB' in ml_model:
                            param_dist = {
                                'n_estimators': [100, 200, 300],
                                'max_depth': [3, 5, 7],
                                'learning_rate': [0.01, 0.05, 0.1],
                                'subsample': [0.8, 0.9, 1.0],
                                'colsample_bytree': [0.8, 0.9, 1.0]
                            }
                        else:  # LightGBM
                            param_dist = {
                                'n_estimators': [100, 200, 300],
                                'max_depth': [3, 5, 7],
                                'learning_rate': [0.01, 0.05, 0.1],
                                'num_leaves': [31, 50, 70]
                            }
                        
                        random_search = RandomizedSearchCV(
                            base_model,
                            param_distributions=param_dist,
                            n_iter=10,
                            scoring='f1' if is_classification else 'r2',
                            cv=3,
                            random_state=random_state,
                            n_jobs=-1
                        )
                        random_search.fit(X_train, y_train)
                        model = random_search.best_estimator_
                        
                        st.success("✅ チューニング完了")
                        st.write("**最適なパラメータ:**", random_search.best_params_)
                    else:
                        model = base_model
                        model.fit(X_train, y_train)
                    
                    # キャリブレーション（分類のみ）
                    if enable_calibration and is_classification:
                        st.info("🔄 確率キャリブレーション実行中...")
                        from sklearn.calibration import CalibratedClassifierCV
                        model = CalibratedClassifierCV(model, cv=3, method='sigmoid')
                        model.fit(X_train, y_train)
                        st.success("✅ キャリブレーション完了")
                
                # 予測
                y_pred = model.predict(X_test)
                
                # 評価
                if is_classification:
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    st.markdown("#### モデル評価")
                    
                    col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
                    col_eval1.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
                    col_eval2.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
                    col_eval3.metric("F1スコア", f"{f1_score(y_test, y_pred):.3f}")
                    if y_pred_proba is not None:
                        col_eval4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.3f}")
                    
                    # 混同行列
                    st.markdown("---")
                    st.markdown("##### 混同行列")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        title="混同行列",
                        labels=dict(x="予測", y="実測"),
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    # 分類レポート
                    st.markdown("---")
                    st.markdown("##### 詳細な分類レポート")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                    
                    # PR曲線（確率予測がある場合）
                    if y_pred_proba is not None:
                        st.markdown("---")
                        st.markdown("##### Precision-Recall曲線")
                        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
                        fig_pr = go.Figure()
                        fig_pr.add_trace(go.Scatter(
                            x=recall_vals,
                            y=precision_vals,
                            mode='lines',
                            name='PR曲線'
                        ))
                        fig_pr.update_layout(
                            title="Precision-Recall曲線",
                            xaxis_title="Recall",
                            yaxis_title="Precision"
                        )
                        st.plotly_chart(fig_pr, use_container_width=True)
                    
                else:  # 回帰
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = np.mean(np.abs(y_test - y_pred))
                    
                    col_eval1, col_eval2, col_eval3 = st.columns(3)
                    col_eval1.metric("R² スコア", f"{r2:.4f}")
                    col_eval2.metric("RMSE", f"{rmse:.4f}")
                    col_eval3.metric("MAE", f"{mae:.4f}")
                    
                    # 予測vs実測
                    st.markdown("---")
                    st.markdown("##### 予測 vs 実測")
                    fig_pred = px.scatter(x=y_test, y=y_pred, title="実測値 vs 予測値",
                                         labels={'x': '実測値', 'y': '予測値'})
                    fig_pred.add_trace(go.Scatter(
                        x=[y_test.min(), y_test.max()],
                        y=[y_test.min(), y_test.max()],
                        mode='lines',
                        name='理想線',
                        line=dict(color='red', dash='dash')
                    ))
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                # 特徴量重要度
                st.markdown("---")
                st.markdown("##### 📊 特徴量重要度")
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        '変数': selected_features,
                        '重要度': model.feature_importances_
                    }).sort_values('重要度', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df,
                        x='重要度',
                        y='変数',
                        orientation='h',
                        title="特徴量重要度（モデルベース）"
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.dataframe(importance_df, use_container_width=True)
                
                # SHAP分析
                if enable_shap and model_category == "高度なモデル（XGBoost/LightGBM）":
                    st.markdown("---")
                    st.markdown("##### 🔍 SHAP分析（解釈可能性）")
                    
                    try:
                        import shap
                        
                        # Calibrated modelの場合は内部モデルを取得
                        if hasattr(model, 'calibrated_classifiers_'):
                            shap_model = model.calibrated_classifiers_[0].estimator
                        else:
                            shap_model = model
                        
                        with st.spinner("SHAP値を計算中..."):
                            explainer = shap.TreeExplainer(shap_model)
                            shap_values = explainer.shap_values(X_test)
                            
                            # バイナリ分類の場合、正クラスのSHAP値を使用
                            if isinstance(shap_values, list):
                                shap_values = shap_values[1]
                            
                            # SHAP重要度
                            shap_importance = pd.DataFrame({
                                '変数': selected_features,
                                'SHAP重要度': np.abs(shap_values).mean(axis=0)
                            }).sort_values('SHAP重要度', ascending=False)
                            
                            st.markdown("###### SHAP特徴量重要度")
                            fig_shap = px.bar(
                                shap_importance.head(10),
                                x='SHAP重要度',
                                y='変数',
                                orientation='h',
                                title="SHAP特徴量重要度（Top 10）"
                            )
                            st.plotly_chart(fig_shap, use_container_width=True)
                            st.dataframe(shap_importance, use_container_width=True)
                            
                            # SHAP summary plot（静的画像）
                            st.markdown("###### SHAP Summary Plot")
                            fig_shap_summary, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values, X_test, feature_names=selected_features, show=False)
                            st.pyplot(fig_shap_summary)
                            plt.close()
                            
                        st.success("✅ SHAP分析完了")
                        
                    except ImportError:
                        st.warning("⚠️ SHAPライブラリがインストールされていません")
                        st.code("pip install shap")
                    except Exception as e:
                        st.warning(f"⚠️ SHAP分析でエラーが発生しました: {e}")
                
                # Permutation Importance（代替）
                if not enable_shap or model_category == "基本モデル":
                    st.markdown("---")
                    st.markdown("##### 🔄 Permutation Importance")
                    
                    try:
                        from sklearn.inspection import permutation_importance
                        
                        with st.spinner("Permutation Importance計算中..."):
                            perm_importance = permutation_importance(
                                model, X_test, y_test,
                                n_repeats=10,
                                random_state=random_state,
                                n_jobs=-1
                            )
                            
                            perm_df = pd.DataFrame({
                                '変数': selected_features,
                                '重要度': perm_importance.importances_mean,
                                '標準偏差': perm_importance.importances_std
                            }).sort_values('重要度', ascending=False)
                            
                            fig_perm = px.bar(
                                perm_df.head(10),
                                x='重要度',
                                y='変数',
                                error_x='標準偏差',
                                orientation='h',
                                title="Permutation Importance（Top 10）"
                            )
                            st.plotly_chart(fig_perm, use_container_width=True)
                            st.dataframe(perm_df, use_container_width=True)
                        
                    except Exception as e:
                        st.warning(f"⚠️ Permutation Importance計算でエラー: {e}")
                
                # ビジネスインサイト
                st.markdown("---")
                st.markdown("##### 💡 ビジネスインサイト")
                
                if hasattr(model, 'feature_importances_'):
                    top_features = importance_df.head(3)['変数'].tolist()
                    st.success(f"**最重要変数（Top 3）**: {', '.join(top_features)}")
                    
                    st.info("""
                    **推奨アクション:**
                    - 上位の重要変数に焦点を当てた施策を検討
                    - 重要度の低い変数は簡略化を検討
                    - 定期的にモデルを再学習して精度を維持
                    """)
                
                # モデルの保存
                st.markdown("---")
                st.markdown("##### 💾 モデルの保存")
                
                import joblib
                from io import BytesIO
                
                model_buffer = BytesIO()
                joblib.dump(model, model_buffer)
                model_buffer.seek(0)
                
                st.download_button(
                    "📥 学習済みモデルをダウンロード",
                    data=model_buffer,
                    file_name="trained_model.pkl",
                    mime="application/octet-stream"
                )
                
                st.caption("※ ダウンロードしたモデルは`joblib.load()`で読み込めます")
    
    # 次のステップへ
    st.markdown("---")
    if st.button("➡️ ステップ5へ進む（解釈とレポート）", type="primary"):
        st.session_state.step = 5
        st.rerun()

# ==================== ステップ5: 解釈とレポート ====================
elif st.session_state.step == 5:
    st.markdown('<div class="step-header">5️⃣ 解釈とレポート生成</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📊 分析結果の解釈とアクションプラン</h4>
    <p>ここでは分析結果を非専門家にもわかりやすく解釈し、具体的なアクションを提案します。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # レポートタイトルと説明
    report_title = st.text_input("レポートタイトル", value="データ分析レポート")
    report_summary = st.text_area(
        "分析サマリ（エグゼクティブサマリ）",
        height=150,
        placeholder="分析の背景、目的、主要な発見を簡潔に記述してください..."
    )
    
    st.markdown("---")
    
    # 主要な発見
    st.subheader("🔍 主要な発見")
    
    num_findings = st.number_input("発見の数", min_value=1, max_value=10, value=3)
    findings = []
    for i in range(num_findings):
        finding = st.text_area(f"発見 {i+1}", key=f"finding_{i}",
                              placeholder="例: 変数Xが目的変数に最も強い正の影響を与えている")
        findings.append(finding)
    
    st.markdown("---")
    
    # 推奨アクション
    st.subheader("💡 推奨アクション")
    
    num_actions = st.number_input("アクションの数", min_value=1, max_value=10, value=3)
    actions = []
    for i in range(num_actions):
        action = st.text_area(f"アクション {i+1}", key=f"action_{i}",
                             placeholder="例: 変数Xの改善に注力する施策を実施する")
        actions.append(action)
    
    st.markdown("---")
    
    # レポート生成
    if st.button("📄 レポート生成（プレビュー）", type="primary"):
        st.markdown("---")
        st.markdown("## 📊 分析レポートプレビュー")
        
        # レポート本文生成
        report_content = f"""
# {report_title}

## エグゼクティブサマリ
{report_summary}

## 分析概要
- **分析日**: {pd.Timestamp.now().strftime('%Y年%m月%d日')}
- **データ**: {st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列
- **目的変数**: {st.session_state.selected_target}
- **説明変数数**: {len(st.session_state.selected_features)}個

## 主要な発見

"""
        for i, finding in enumerate(findings, 1):
            if finding:
                report_content += f"{i}. {finding}\n\n"
        
        report_content += """
## 推奨アクション

"""
        for i, action in enumerate(actions, 1):
            if action:
                report_content += f"{i}. {action}\n\n"
        
        report_content += """
## 次のステップ
1. 提案されたアクションの優先順位付け
2. 実行計画の策定（タイムライン、担当者、予算）
3. KPI設定と効果測定の仕組み構築
4. 定期的なモニタリングと改善サイクルの確立

---
*本レポートはData Science Workflow Studioにより生成されました*
"""
        
        st.markdown(report_content)
        
        # ダウンロードボタン
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Markdownダウンロード
            md_bytes = report_content.encode('utf-8')
            st.download_button(
                "📥 レポートをダウンロード（Markdown）",
                data=md_bytes,
                file_name="analysis_report.md",
                mime="text/markdown"
            )
        
        with col2:
            # CSV結果ダウンロード
            if st.session_state.processed_df is not None:
                csv_bytes = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 処理済みデータをダウンロード（CSV）",
                    data=csv_bytes,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
    
    # 分析の完了
    st.markdown("---")
    st.markdown("""
    <div class="success-box">
    <h3>🎉 分析が完了しました！</h3>
    <p>生成されたレポートを関係者と共有し、推奨アクションの実装を検討してください。</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_final1, col_final2, col_final3 = st.columns(3)
    
    with col_final1:
        if st.button("🔄 新しい分析を開始", type="secondary"):
            # セッションステートをリセット
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.step = 1
            st.rerun()
    
    with col_final2:
        if st.button("📊 ステップ4に戻る（追加分析）", type="secondary"):
            st.session_state.step = 4
            st.rerun()
    
    with col_final3:
        if st.button("🔧 ステップ3に戻る（データ再処理）", type="secondary"):
            st.session_state.step = 3
            st.rerun()

# フッター
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p><strong>Data Science Workflow Studio</strong></p>
    <p>実務に特化した直感的なデータ分析プラットフォーム</p>
    <p style="font-size: 0.8rem;">問題定義 → データ理解 → 前処理 → モデリング → 解釈 → アクション</p>
</div>
""", unsafe_allow_html=True)

# サイドバーにヘルプ情報
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📚 ヘルプ")
    with st.expander("💡 使い方のヒント"):
        st.markdown("""
        **ステップ1: 問題定義**
        - 何のための分析か明確に
        - 期待する成果を具体的に
        
        **ステップ2: データ理解**
        - 変数名をわかりやすく編集
        - 目的変数と説明変数を選択
        
        **ステップ3: データ前処理**
        - 欠損値・外れ値を処理
        - 必要な変換を適用
        
        **ステップ4: モデリング**
        - 目的に合った分析を選択
        - 結果を解釈
        
        **ステップ5: レポート**
        - 発見とアクションを記載
        - 関係者と共有
        """)
    
    with st.expander("🔧 トラブルシューティング"):
        st.markdown("""
        **エラーが出た場合**
        1. データに欠損値がないか確認
        2. 数値型の変数が選択されているか確認
        3. ステップ3で前処理を実行
        
        **予期しない結果の場合**
        1. データの品質を再確認
        2. 外れ値の影響を検討
        3. 変数選択を見直す
        
        **パフォーマンスが遅い場合**
        1. データサイズを確認
        2. 説明変数の数を減らす
        3. サンプリングを検討
        """)
    
    with st.expander("📖 統計用語集"):
        st.markdown("""
        **R²（決定係数）**: モデルの当てはまりの良さ（0-1）
        
        **RMSE**: 予測誤差の大きさ
        
        **p値**: 統計的有意性（< 0.05で有意）
        
        **VIF**: 多重共線性の指標（> 10で問題）
        
        **相関係数**: 2変数間の関係の強さ（-1 to 1）
        """)

# EOF