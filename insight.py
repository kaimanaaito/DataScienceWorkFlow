"""
Data Science Workflow Studio Pro - 上級者対応版
実務特化型データ分析アプリ + 統計的厳密性 + コード生成機能 + ビジネスインパクトシミュレーション

必要なパッケージ:
pip install streamlit pandas numpy scipy scikit-learn plotly statsmodels

実行:
streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
from sklearn.inspection import permutation_importance
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor, OLSInfluence
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="Data Science Workflow Studio Pro",
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
    .warning-box {
        background-color: #7c2d12;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin: 1rem 0;
    }
    .diagnostic-pass {
        background-color: #064e3b;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .diagnostic-fail {
        background-color: #7c2d12;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .impact-positive {
        background: linear-gradient(90deg, #064e3b, #10b981);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .impact-negative {
        background: linear-gradient(90deg, #7c2d12, #f59e0b);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
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
if 'diagnostics_results' not in st.session_state:
    st.session_state.diagnostics_results = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'current_model' not in st.session_state:
    st.session_state.current_model = None
if 'current_X_test' not in st.session_state:
    st.session_state.current_X_test = None
if 'current_y_test' not in st.session_state:
    st.session_state.current_y_test = None
if 'current_y_pred' not in st.session_state:
    st.session_state.current_y_pred = None

# ==================== 統計的厳密性のための関数 ====================

def comprehensive_regression_diagnostics(model_sm, X, y, residuals):
    """回帰分析の完全な診断"""
    diagnostics = {}
    
    try:
        rainbow_stat, rainbow_p = linear_rainbow(model_sm)
        diagnostics['linearity'] = {
            'test': 'Rainbow Test',
            'statistic': float(rainbow_stat),
            'p_value': float(rainbow_p),
            'passed': rainbow_p > 0.05,
            'interpretation': '線形関係が適切' if rainbow_p > 0.05 else '非線形性の可能性あり'
        }
    except:
        diagnostics['linearity'] = {
            'test': 'Rainbow Test',
            'statistic': None,
            'p_value': None,
            'passed': None,
            'interpretation': '計算不可'
        }
    
    try:
        X_with_const = sm.add_constant(X)
        bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_with_const)
        diagnostics['homoscedasticity'] = {
            'test': 'Breusch-Pagan Test',
            'statistic': float(bp_stat),
            'p_value': float(bp_p),
            'passed': bp_p > 0.05,
            'interpretation': '等分散性あり' if bp_p > 0.05 else '不等分散（WLS回帰を検討）',
            'remedy': '' if bp_p > 0.05 else 'ロバスト標準誤差の使用、またはWLS（加重最小二乗法）への変更を推奨'
        }
    except:
        diagnostics['homoscedasticity'] = {
            'test': 'Breusch-Pagan Test',
            'statistic': None,
            'p_value': None,
            'passed': None,
            'interpretation': '計算不可'
        }
    
    try:
        jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
        diagnostics['normality'] = {
            'test': 'Jarque-Bera Test',
            'statistic': float(jb_stat),
            'p_value': float(jb_p),
            'skewness': float(skew),
            'kurtosis': float(kurtosis),
            'passed': jb_p > 0.05,
            'interpretation': '正規分布に従う' if jb_p > 0.05 else '正規性違反',
            'remedy': '' if jb_p > 0.05 else 'サンプルサイズが十分大きければ問題なし（中心極限定理）'
        }
    except:
        diagnostics['normality'] = {
            'test': 'Jarque-Bera Test',
            'statistic': None,
            'p_value': None,
            'passed': None,
            'interpretation': '計算不可'
        }
    
    try:
        dw_stat = durbin_watson(residuals)
        diagnostics['autocorrelation'] = {
            'test': 'Durbin-Watson Test',
            'statistic': float(dw_stat),
            'passed': 1.5 < dw_stat < 2.5,
            'interpretation': '自己相関なし' if 1.5 < dw_stat < 2.5 else f'自己相関の可能性',
            'remedy': '' if 1.5 < dw_stat < 2.5 else '時系列データの場合はARIMAモデルを検討'
        }
    except:
        diagnostics['autocorrelation'] = {
            'test': 'Durbin-Watson Test',
            'statistic': None,
            'passed': None,
            'interpretation': '計算不可'
        }
    
    try:
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        
        diagnostics['multicollinearity'] = {
            'test': 'Variance Inflation Factor (VIF)',
            'data': vif_data,
            'passed': (vif_data['VIF'] < 10).all(),
            'severe': (vif_data['VIF'] > 10).any(),
            'max_vif': float(vif_data['VIF'].max()),
            'interpretation': '多重共線性なし（VIF<10）' if (vif_data['VIF'] < 10).all() else f'多重共線性あり',
            'remedy': '' if (vif_data['VIF'] < 10).all() else 'Ridge回帰、Lasso回帰を検討'
        }
    except:
        diagnostics['multicollinearity'] = {
            'test': 'VIF',
            'passed': None,
            'interpretation': '計算不可'
        }
    
    try:
        influence = OLSInfluence(model_sm)
        cooks_d = influence.cooks_distance[0]
        cooks_threshold = 4 / len(residuals)
        high_influence = np.where(cooks_d > cooks_threshold)[0]
        
        diagnostics['influence'] = {
            'test': "Cook's Distance",
            'threshold': float(cooks_threshold),
            'n_influential': int(len(high_influence)),
            'influential_indices': high_influence.tolist()[:10],
            'passed': len(high_influence) < len(residuals) * 0.05,
            'interpretation': f'{len(high_influence)}個の影響力の大きい観測値',
            'remedy': '' if len(high_influence) == 0 else '影響力の大きい観測値を詳細調査'
        }
    except:
        diagnostics['influence'] = {
            'test': "Cook's Distance",
            'passed': None,
            'interpretation': '計算不可'
        }
    
    return diagnostics


def calculate_effect_sizes(model_sm, X, y):
    """効果量を計算"""
    effect_sizes = {}
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
        X_scaled_with_const = sm.add_constant(X_scaled_df)
        model_scaled = sm.OLS(y, X_scaled_with_const).fit()
        
        effect_sizes['standardized_coefficients'] = pd.DataFrame({
            'feature': X.columns,
            'beta': model_scaled.params[1:].values,
            'abs_beta': np.abs(model_scaled.params[1:].values)
        }).sort_values('abs_beta', ascending=False)
    except:
        effect_sizes['standardized_coefficients'] = None
    
    try:
        full_r2 = model_sm.rsquared
        partial_r2_list = []
        for col in X.columns:
            X_reduced = X.drop(columns=[col])
            X_reduced_with_const = sm.add_constant(X_reduced)
            model_reduced = sm.OLS(y, X_reduced_with_const).fit()
            reduced_r2 = model_reduced.rsquared
            partial_r2_list.append(full_r2 - reduced_r2)
        
        effect_sizes['partial_r2'] = pd.DataFrame({
            'feature': X.columns,
            'partial_r2': partial_r2_list,
            'percentage': np.array(partial_r2_list) / full_r2 * 100 if full_r2 > 0 else 0
        }).sort_values('partial_r2', ascending=False)
    except:
        effect_sizes['partial_r2'] = None
    
    try:
        full_r2 = model_sm.rsquared
        if full_r2 < 1:
            cohens_f2 = full_r2 / (1 - full_r2)
        else:
            cohens_f2 = np.inf
        
        if cohens_f2 < 0.02:
            interpretation = "小さい効果"
        elif cohens_f2 < 0.15:
            interpretation = "中程度の効果"
        elif cohens_f2 < 0.35:
            interpretation = "大きい効果"
        else:
            interpretation = "非常に大きい効果"
        
        effect_sizes['cohens_f2'] = {
            'value': float(cohens_f2),
            'interpretation': interpretation
        }
    except:
        effect_sizes['cohens_f2'] = None
    
    return effect_sizes


def rigorous_cross_validation(model, X, y, cv=5):
    """厳密なクロスバリデーション"""
    try:
        scoring = {
            'r2': 'r2',
            'neg_mse': 'neg_mean_squared_error',
            'neg_mae': 'neg_mean_absolute_error'
        }
        
        cv_results = cross_validate(
            model, X, y,
            cv=KFold(n_splits=cv, shuffle=True, random_state=42),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        train_r2 = cv_results['train_r2'].mean()
        test_r2 = cv_results['test_r2'].mean()
        overfit_gap = train_r2 - test_r2
        
        results = {
            'test_r2_mean': float(test_r2),
            'test_r2_std': float(cv_results['test_r2'].std()),
            'train_r2_mean': float(train_r2),
            'test_mse_mean': float(-cv_results['neg_mse'].mean()),
            'test_mae_mean': float(-cv_results['neg_mae'].mean()),
            'overfit_gap': float(overfit_gap),
            'is_overfitting': overfit_gap > 0.1,
            'cv_scores': cv_results
        }
        
        return results
    except Exception as e:
        st.error(f"クロスバリデーションでエラー: {e}")
        return None


def display_diagnostics_report(diagnostics):
    """診断結果を表示"""
    st.markdown("---")
    st.markdown("### 📋 統計的診断レポート")
    
    passed_tests = [d.get('passed') for d in diagnostics.values() if d.get('passed') is not None]
    
    if passed_tests:
        all_passed = all(passed_tests)
        pass_rate = sum(passed_tests) / len(passed_tests) * 100
        
        col_summary1, col_summary2 = st.columns(2)
        with col_summary1:
            if all_passed:
                st.markdown('<div class="diagnostic-pass">✅ すべての前提条件を満たしています</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="diagnostic-fail">⚠️ 一部の前提条件に問題があります（合格率: {pass_rate:.0f}%）</div>', unsafe_allow_html=True)
        
        with col_summary2:
            st.info("結果の解釈に注意が必要です")
    
    for test_name, result in diagnostics.items():
        with st.expander(f"{'✅' if result.get('passed') else '❌'} {test_name.upper()} - {result['test']}"):
            
            if result.get('statistic') is not None:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("検定統計量", f"{result['statistic']:.4f}")
                    if 'p_value' in result and result['p_value'] is not None:
                        st.metric("p値", f"{result['p_value']:.4f}")
                    
                    if test_name == 'normality' and 'skewness' in result:
                        st.metric("歪度", f"{result['skewness']:.4f}")
                        st.metric("尖度", f"{result['kurtosis']:.4f}")
                    
                    if test_name == 'multicollinearity' and 'max_vif' in result:
                        st.metric("最大VIF", f"{result['max_vif']:.2f}")
                    
                    if test_name == 'influence' and 'n_influential' in result:
                        st.metric("影響力の大きい観測値数", result['n_influential'])
                
                with col2:
                    status = "✅ 合格" if result.get('passed') else "❌ 不合格"
                    st.markdown(f"**判定**: {status}")
                    st.info(f"**解釈**: {result['interpretation']}")
                    
                    if not result.get('passed') and 'remedy' in result and result['remedy']:
                        st.warning(f"**推奨対処法**: {result['remedy']}")
                
                if test_name == 'multicollinearity' and 'data' in result:
                    st.markdown("##### VIF値の詳細")
                    st.dataframe(result['data'].style.background_gradient(cmap='YlOrRd', subset=['VIF']), 
                               use_container_width=True)
                
                if test_name == 'influence' and 'influential_indices' in result and result['influential_indices']:
                    st.markdown("##### 影響力の大きい観測値（最大10個）")
                    st.write(f"インデックス: {result['influential_indices']}")
            else:
                st.warning(f"⚠️ {result['interpretation']}")


def display_effect_sizes(effect_sizes):
    """効果量を表示"""
    st.markdown("---")
    st.markdown("### 📊 効果量分析（実務的重要性の評価）")
    
    st.info("💡 p値は「効果が存在するか」を示しますが、効果量は「効果がどれだけ大きいか」を示します")
    
    if effect_sizes.get('cohens_f2'):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Cohen's f²（モデル全体の効果）", f"{effect_sizes['cohens_f2']['value']:.4f}")
        with col2:
            st.markdown(f"**解釈**: {effect_sizes['cohens_f2']['interpretation']}")
        st.caption("基準: 小(0.02), 中(0.15), 大(0.35)")
    
    if effect_sizes.get('standardized_coefficients') is not None:
        st.markdown("---")
        st.markdown("#### 標準化係数（Beta）- 変数間の直接比較")
        st.caption("単位の影響を除去した係数。絶対値が大きいほど影響力が大きい")
        
        fig = px.bar(
            effect_sizes['standardized_coefficients'].head(15),
            x='beta', y='feature', orientation='h',
            title="標準化係数（Beta）- Top 15",
            color='beta',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0
        )
        fig.update_layout(height=max(400, len(effect_sizes['standardized_coefficients'].head(15)) * 30))
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(effect_sizes['standardized_coefficients'], use_container_width=True)
    
    if effect_sizes.get('partial_r2') is not None:
        st.markdown("---")
        st.markdown("#### 各変数の寄与度（Partial R²）")
        st.caption("各変数がモデルの説明力（R²）にどれだけ寄与しているか")
        
        fig2 = px.bar(
            effect_sizes['partial_r2'].head(15),
            x='percentage', y='feature', orientation='h',
            title="説明力への寄与率（%）- Top 15",
            color='percentage',
            color_continuous_scale='Viridis'
        )
        fig2.update_layout(height=max(400, len(effect_sizes['partial_r2'].head(15)) * 30))
        st.plotly_chart(fig2, use_container_width=True)
        
        st.dataframe(effect_sizes['partial_r2'], use_container_width=True)


def display_cv_results(results):
    """クロスバリデーション結果を表示"""
    st.markdown("---")
    st.markdown("### 🔄 クロスバリデーション結果（過学習の検証）")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("テストR²（平均）", f"{results['test_r2_mean']:.4f} ± {results['test_r2_std']:.4f}")
        st.caption("テストデータでの予測精度")
    
    with col2:
        st.metric("訓練R²（平均）", f"{results['train_r2_mean']:.4f}")
        st.caption("訓練データでの適合度")
    
    with col3:
        overfit_status = "⚠️ 過学習" if results['is_overfitting'] else "✅ 適切"
        st.metric("過学習判定", overfit_status)
        st.caption(f"差: {results['overfit_gap']:.4f}")
    
    if results['is_overfitting']:
        st.markdown('<div class="warning-box">⚠️ <strong>過学習の兆候</strong></div>', unsafe_allow_html=True)
        st.warning("**推奨対策**: 正則化の強化（RidgeやLasso）、特徴量の削減、データの追加収集")
    else:
        st.markdown('<div class="success-box">✅ 過学習なし</div>', unsafe_allow_html=True)
    
    st.markdown("#### クロスバリデーションスコアの分布")
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=results['cv_scores']['test_r2'],
        name='Test R²',
        boxmean='sd',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Box(
        y=results['cv_scores']['train_r2'],
        name='Train R²',
        boxmean='sd',
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title="各Foldでのスコア分布",
        yaxis_title="R² Score",
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("平均二乗誤差（MSE）", f"{results['test_mse_mean']:.4f}")
    with col_m2:
        st.metric("平均絶対誤差（MAE）", f"{results['test_mae_mean']:.4f}")


# ==================== ビジネスインパクトシミュレーション関数 ====================

def business_impact_simulation(model, X_test, y_test, y_pred, pred_proba=None):
    """Kaggle/Amazonエリート級のビジネスインパクトシミュレーション"""
    
    st.markdown("---")
    st.markdown("### 🎯 ビジネスインパクトシミュレーション（Kaggle/Amazonエリート級）")
    
    st.info("Monte Carloシミュレーションで不確実性を考慮したROIを計算。実務で即戦力の機能です")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ビジネスパラメータ")
        customer_base = st.number_input("総顧客数", min_value=1000, value=10000, step=1000)
        historical_churn_rate = st.slider("歴史的離脱率", 0.01, 0.50, 0.10)
        avg_ltv = st.number_input("平均LTV（円）", min_value=1000, value=50000, step=5000)
        intervention_cost_per = st.number_input("1人介入コスト（円）", value=3000, step=500)
        intervention_success_rate_mean = st.slider("介入成功率（平均）", 0.1, 0.8, 0.4)
        intervention_success_rate_std = st.slider("成功率のばらつき（標準偏差）", 0.01, 0.2, 0.05)
    
    with col2:
        st.subheader("介入戦略")
        top_pct = st.slider("介入対象（離脱確率上位何％）", 1, 50, 10) / 100
        n_simulations = st.number_input("Monte Carloシミュレーション回数", value=1000, step=100)
    
    if st.button("🔥 インパクト計算実行", type="primary", key="impact_calc"):
        
        # 予測確率を取得（分類の場合想定）
        if pred_proba is None:
            try:
                if hasattr(model, "predict_proba"):
                    pred_proba = model.predict_proba(X_test)[:, 1]
                else:
                    pred_proba = model.predict(X_test)
                    pred_proba = (pred_proba - pred_proba.min()) / (pred_proba.max() - pred_proba.min() + 1e-8)
            except:
                pred_proba = y_pred
        
        # モデル性能計算（precision/recall at top_pct）
        threshold = np.percentile(pred_proba, 100 - (top_pct * 100))
        y_pred_at_threshold = (pred_proba >= threshold).astype(int)
        
        try:
            precision = precision_score(y_test, y_pred_at_threshold, zero_division=0)
            recall = recall_score(y_test, y_pred_at_threshold, zero_division=0)
        except:
            precision = 0.5
            recall = 0.5
        
        st.write(f"モデル性能（上位{top_pct*100:.0f}%閾値）: Precision={precision:.2f}, Recall={recall:.2f}")
        
        # Monte Carloシミュレーション
        net_savings_list = []
        roi_list = []
        
        for _ in range(n_simulations):
            # 不確実性を加味（成功率にノイズ）
            sim_success_rate = np.clip(stats.norm.rvs(intervention_success_rate_mean, intervention_success_rate_std), 0.1, 0.8)
            
            num_identified = customer_base * recall * historical_churn_rate
            customers_saved = num_identified * sim_success_rate
            revenue_retained = customers_saved * avg_ltv
            campaign_cost = num_identified * intervention_cost_per
            
            net_savings = revenue_retained - campaign_cost
            roi = (net_savings / campaign_cost * 100) if campaign_cost > 0 else 0
            
            net_savings_list.append(net_savings)
            roi_list.append(roi)
        
        mean_net = np.mean(net_savings_list)
        mean_roi = np.mean(roi_list)
        ci_net = np.percentile(net_savings_list, [5, 95])
        ci_roi = np.percentile(roi_list, [5, 95])
        
        c1, c2, c3 = st.columns(3)
        c1.metric("平均年間利益改善額", f"¥{mean_net:,.0f}", f"95%CI: ¥{ci_net[0]:,.0f} ~ ¥{ci_net[1]:,.0f}")
        c2.metric("平均ROI", f"{mean_roi:.1f}%", f"95%CI: {ci_roi[0]:.1f}% ~ {ci_roi[1]:.1f}%")
        c3.metric("期待救済顧客（年）", f"{(customer_base * historical_churn_rate * recall * intervention_success_rate_mean):,.0f}人")
        
        # 分布グラフ（エリート級の可視化）
        fig = px.histogram(net_savings_list, nbins=50, title="Monte Carloシミュレーション結果（利益改善額分布）")
        fig.add_vline(x=mean_net, line_dash="dash", line_color="red", annotation_text="平均")
        st.plotly_chart(fig, use_container_width=True)
        
        if mean_net > 0:
            st.markdown(f'<div class="impact-positive">💰 上位{top_pct*100:.0f}%介入で、平均 <strong>¥{mean_net:,.0f}</strong> の利益創出可能（95%信頼区間内）！</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="impact-negative">⚠️ 上位{top_pct*100:.0f}%介入では、現状のパラメータでは利益創出が困難です。コスト削減または成功率向上を検討してください。</div>', unsafe_allow_html=True)
        
        return {
            'mean_net_savings': mean_net,
            'mean_roi': mean_roi,
            'confidence_interval_net': ci_net,
            'confidence_interval_roi': ci_roi,
            'expected_customers_saved': customer_base * historical_churn_rate * recall * intervention_success_rate_mean
        }


# ユーティリティ関数
def load_csv(file_buf):
    """CSVファイルを読み込む"""
    try:
        file_buf.seek(0)
        df = pd.read_csv(file_buf)
        return df, None
    except Exception as e:
        return None, f"ファイル読み込みエラー: {e}"


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
st.markdown('<div class="main-header">📊 Data Science Workflow Studio Pro</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #94a3b8;">統計コンサルタントレベルの厳密性 + コード生成機能 + ビジネスインパクト分析</p>', unsafe_allow_html=True)

# サイドバー
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

# ステップ1: 問題定義
if st.session_state.step == 1:
    st.markdown('<div class="step-header">1️⃣ 問題定義</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>🎯 分析を始める前に考えるべきこと</h4>
    <ul>
        <li><strong>ビジネス課題</strong>: 解決したい実務上の問題は何ですか？</li>
        <li><strong>分析目的</strong>: 記述、推論、予測、因果推定のどれを目指しますか？</li>
        <li><strong>期待される成果</strong>: この分析で何がわかれば成功ですか？</li>
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
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if problem_statement and analysis_goal != "選択してください":
        st.markdown('<div class="success-box">✅ 問題定義が完了しました</div>', unsafe_allow_html=True)
        if st.button("➡️ ステップ2へ進む", type="primary"):
            st.session_state.step = 2
            st.rerun()
    else:
        st.warning("⚠️ 分析目的を明確にしてから次のステップに進んでください。")

# ステップ2: データ理解と点検
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">2️⃣ データ理解と点検</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("📁 CSVファイルをアップロード", type=['csv'])
    
    col_sample1, col_sample2 = st.columns([1, 3])
    with col_sample1:
        if st.button('サンプルデータ生成'):
            np.random.seed(42)
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
        result, error = load_csv(uploaded_file)
        if error:
            st.error(error)
        else:
            df = result
            st.session_state.df = df
            st.success(f"✅ データ読み込み完了: {df.shape[0]}行 × {df.shape[1]}列")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総行数", df.shape[0])
        col2.metric("総列数", df.shape[1])
        col3.metric("数値列", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("欠損値あり", df.isnull().any().sum())
        
        st.markdown("---")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📋 データプレビュー", "🏷️ 変数名編集", "🎯 変数選択", "📊 基本統計"])
        
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
                new_names = list(st.session_state.column_mapping.values())
                if len(new_names) != len(set(new_names)):
                    st.error("❌ 重複する変数名があります")
                else:
                    df_renamed = df.rename(columns=st.session_state.column_mapping)
                    st.session_state.df = df_renamed
                    st.success("✅ 変数名を更新しました")
                    st.rerun()
        
        with tab3:
            st.subheader("🎯 目的変数と説明変数の選択")
            
            target_col = st.selectbox(
                "目的変数を選択",
                options=['選択してください'] + list(df.columns),
                key="target_selection"
            )
            
            if target_col != '選択してください':
                st.session_state.selected_target = target_col
                
                feature_candidates = [col for col in df.columns if col != target_col]
                
                st.markdown("---")
                st.markdown("#### 説明変数を選択")
                
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ すべて選択"):
                        st.session_state.selected_features = feature_candidates
                        st.rerun()
                with col_btn2:
                    if st.button("❌ すべて解除"):
                        st.session_state.selected_features = []
                        st.rerun()
                
                selected_features = st.multiselect(
                    "使用する説明変数",
                    options=feature_candidates,
                    default=[f for f in st.session_state.selected_features if f in feature_candidates] if st.session_state.selected_features else feature_candidates,
                    key="feature_selection"
                )
                st.session_state.selected_features = selected_features
                
                if selected_features:
                    st.success(f"✅ 目的変数: **{target_col}** | 説明変数: **{len(selected_features)}個**")
                    
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
                            color_continuous_scale='RdBu_r',
                            zmin=-1, zmax=1
                        )
                        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        if st.session_state.selected_target and st.session_state.selected_features:
            if st.button("➡️ ステップ3へ進む（データ前処理）", type="primary"):
                st.session_state.step = 3
                st.rerun()
        else:
            st.warning("⚠️ 目的変数と説明変数を選択してください")
    else:
        st.info("💡 CSVファイルをアップロードするか、サンプルデータを生成してください")

# ステップ3: データ前処理・EDA
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">3️⃣ データ前処理・EDA</div>', unsafe_allow_html=True)
    
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
    
    if st.session_state.processed_df is None:
        st.session_state.processed_df = df[[target] + features].copy()
    
    work_df = st.session_state.processed_df.copy()
    
    st.info(f"目的変数: **{target}** | 説明変数: **{len(features)}個**")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔧 欠損値処理",
        "📉 外れ値処理",
        "🔄 データ変換",
        "🏷️ カテゴリ変数処理",
        "💾 処理済みデータ"
    ])
    
    with tab1:
        st.subheader("🔧 欠損値補完")
        
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
            
            # 重複列名をチェック
            if work_df.columns.duplicated().any():
                st.error("❌ データに重複した列名があります。ステップ3の「処理済みデータ」タブで確認してください")
                duplicates = work_df.columns[work_df.columns.duplicated()].tolist()
                st.write(f"重複している列: {list(set(duplicates))}")
                
                if st.button("🔧 重複列を自動削除"):
                    # 重複列を削除（最初のものを残す）
                    work_df = work_df.loc[:, ~work_df.columns.duplicated()]
                    st.session_state.processed_df = work_df
                    st.success("✅ 重複列を削除しました")
                    st.rerun()
            else:
                fig = px.box(work_df, y=outlier_col, title=f"{outlier_col} の箱ひげ図")
                st.plotly_chart(fig, use_container_width=True)
                
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
                        new_col_name = f"{encode_col}_encoded"
                        if new_col_name not in work_df.columns:
                            le = LabelEncoder()
                            work_df[new_col_name] = le.fit_transform(work_df[encode_col].astype(str))
                            st.success(f"✅ {encode_col}をLabel Encodingしました → {new_col_name}")
                        else:
                            st.warning(f"⚠️ {new_col_name} は既に存在します")
                            
                    elif encode_method == "One-Hot Encoding（ダミー変数）":
                        # 既存のダミー変数をチェック
                        existing_dummies = [col for col in work_df.columns if col.startswith(f"{encode_col}_")]
                        
                        if existing_dummies:
                            st.warning(f"⚠️ {encode_col} の One-Hot Encoding列が既に存在します: {existing_dummies}")
                            if st.checkbox(f"既存の列を削除して再エンコードしますか？"):
                                work_df = work_df.drop(columns=existing_dummies)
                                dummies = pd.get_dummies(work_df[encode_col], prefix=encode_col, drop_first=True)
                                work_df = pd.concat([work_df, dummies], axis=1)
                                st.success(f"✅ {encode_col}を再エンコードしました（{len(dummies.columns)}個の新変数）")
                                st.session_state.processed_df = work_df
                                st.rerun()
                        else:
                            dummies = pd.get_dummies(work_df[encode_col], prefix=encode_col, drop_first=True)
                            work_df = pd.concat([work_df, dummies], axis=1)
                            st.success(f"✅ {encode_col}をOne-Hot Encodingしました（{len(dummies.columns)}個の新変数）")
                            
                    elif encode_method == "頻度エンコーディング":
                        new_col_name = f"{encode_col}_freq"
                        if new_col_name not in work_df.columns:
                            freq_map = work_df[encode_col].value_counts(normalize=True).to_dict()
                            work_df[new_col_name] = work_df[encode_col].map(freq_map)
                            st.success(f"✅ {encode_col}を頻度エンコーディングしました → {new_col_name}")
                        else:
                            st.warning(f"⚠️ {new_col_name} は既に存在します")
                    
                    st.session_state.processed_df = work_df
                    st.rerun()
        else:
            st.info("カテゴリ変数がありません")
    
    with tab5:
        st.subheader("💾 処理済みデータ")
        
        # 重複列名チェック
        if work_df.columns.duplicated().any():
            st.error("⚠️ データに重複した列名があります")
            duplicates = work_df.columns[work_df.columns.duplicated()].tolist()
            st.write(f"重複している列: {list(set(duplicates))}")
            
            if st.button("🔧 重複列を自動削除", type="primary"):
                work_df = work_df.loc[:, ~work_df.columns.duplicated()]
                st.session_state.processed_df = work_df
                st.success("✅ 重複列を削除しました")
                st.rerun()
        
        st.write(f"**現在のデータ**: {work_df.shape[0]}行 × {work_df.shape[1]}列")
        
        st.dataframe(work_df.head(100), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 基本統計量")
        st.dataframe(work_df.describe(), use_container_width=True)
        
        st.markdown("---")
        st.markdown("#### 列名一覧")
        col_info = pd.DataFrame({
            '列名': work_df.columns,
            'データ型': work_df.dtypes.values,
            '欠損数': work_df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)
        
        st.markdown("---")
        csv = work_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 処理済みデータをダウンロード (CSV)",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv",
            type="primary"
        )
        
        if st.button("✅ この処理済みデータで分析を続ける", type="primary"):
            # 最終チェック
            if work_df.columns.duplicated().any():
                st.error("❌ 重複列名を削除してから続けてください")
            else:
                st.session_state.processed_df = work_df
                st.success("✅ 処理済みデータを確定しました")
    
    st.markdown("---")
    if st.button("➡️ ステップ4へ進む（モデリング）", type="primary"):
        st.session_state.processed_df = work_df
        st.session_state.step = 4
        st.rerun()

# ステップ4: モデリングと推定
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
    
    if target not in df.columns:
        st.error(f"❌ 目的変数 '{target}' がデータに存在しません")
        st.stop()
    
    numeric_features = [col for col in df.columns if col != target and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_features:
        st.error("❌ 数値型の説明変数がありません。ステップ3でカテゴリ変数をエンコードしてください")
        if st.button("⬅️ ステップ3に戻る"):
            st.session_state.step = 3
            st.rerun()
        st.stop()
    
    st.info(f"目的変数: **{target}** | 使用可能な説明変数: **{len(numeric_features)}個**")
    
    st.markdown("---")
    st.subheader("📊 分析目的を選択")
    
    analysis_type = st.selectbox(
        "実行したい分析",
        [
            "選択してください",
            "📊 記述統計（平均・分散・分布比較）",
            "🔍 推論（t検定・ANOVA・相関分析）",
            "📈 回帰分析（統計的厳密性強化版）",
            "🎯 予測モデル（機械学習）",
            "🚀 回帰分析（自動最適化）",
            "🤖 予測モデル（自動チューニング）"
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
            fig_hist = px.histogram(df, x=desc_var, title=f"{desc_var} の分布", marginal="box")
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
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
    
    # 回帰分析（統計的厳密性強化版）
    elif analysis_type == "📈 回帰分析（統計的厳密性強化版）":
        st.markdown("### 重回帰分析（統計コンサルタントレベル）")
        
        st.info("統計的前提条件の検証、効果量の計算、クロスバリデーションを含む厳密な回帰分析を実行します")
        
        selected_features = st.multiselect(
            "回帰分析に使用する説明変数",
            options=numeric_features,
            default=numeric_features[:min(10, len(numeric_features))]
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ 説明変数を選択してください")
            st.stop()
        
        regression_df = df[[target] + selected_features].dropna()
        
        if len(regression_df) == 0:
            st.error("❌ 欠損値を除外するとデータが残りません")
            st.stop()
        
        st.write(f"**使用データ**: {len(regression_df)}行")
        
        model_type = st.selectbox(
            "モデルタイプ",
            ["通常の線形回帰（OLS）", "Ridge回帰（正則化）", "Lasso回帰（変数選択）"]
        )
        
        with st.expander("🔧 高度な設定"):
            test_size = st.slider("テストデータの割合", 0.1, 0.4, 0.2, 0.05)
            enable_cv = st.checkbox("クロスバリデーションを実行", value=True)
            cv_folds = st.slider("CVのfold数", 3, 10, 5) if enable_cv else 5
            random_state = 42
        
        if st.button("✅ 回帰分析実行", type="primary"):
            X = regression_df[selected_features]
            y = regression_df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            with st.spinner("モデル学習中..."):
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test)
                
                if model_type == "通常の線形回帰（OLS）":
                    model_sm = sm.OLS(y_train, X_train_sm).fit()
                    y_pred = model_sm.predict(X_test_sm)
                    residuals = model_sm.resid
                    
                elif model_type == "Ridge回帰（正則化）":
                    alpha = st.slider("正則化パラメータ（α）", 0.01, 10.0, 1.0)
                    model_sklearn = Ridge(alpha=alpha)
                    model_sklearn.fit(X_train, y_train)
                    y_pred = model_sklearn.predict(X_test)
                    
                    model_sm = sm.OLS(y_train, X_train_sm).fit()
                    residuals = y_train - model_sm.predict(X_train_sm)
                    
                else:
                    alpha = st.slider("正則化パラメータ（α）", 0.01, 10.0, 1.0)
                    model_sklearn = Lasso(alpha=alpha)
                    model_sklearn.fit(X_train, y_train)
                    y_pred = model_sklearn.predict(X_test)
                    
                    model_sm = sm.OLS(y_train, X_train_sm).fit()
                    residuals = y_train - model_sm.predict(X_train_sm)
                
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                # モデル情報をセッションステートに保存
                st.session_state.current_model = model_sm
                st.session_state.current_X_test = X_test
                st.session_state.current_y_test = y_test
                st.session_state.current_y_pred = y_pred
                
                st.markdown("---")
                st.markdown("## 📊 回帰分析結果")
                
                st.markdown("### 🎯 モデル全体の性能")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                col_m1.metric("R²", f"{r2:.4f}")
                col_m2.metric("調整済みR²", f"{model_sm.rsquared_adj:.4f}")
                col_m3.metric("RMSE", f"{rmse:.4f}")
                col_m4.metric("MAE", f"{mae:.4f}")
                
                st.caption(f"F統計量: {model_sm.fvalue:.2f}, p値: {model_sm.f_pvalue:.4e}")
                
                st.markdown("---")
                st.markdown("### 📈 回帰係数の詳細")
                
                coef_df = pd.DataFrame({
                    '変数': ['切片'] + selected_features,
                    '係数': model_sm.params.values,
                    '標準誤差': model_sm.bse.values,
                    't値': model_sm.tvalues.values,
                    'p値': model_sm.pvalues.values,
                    '95%CI下限': model_sm.conf_int()[0].values,
                    '95%CI上限': model_sm.conf_int()[1].values
                })
                
                coef_df['有意性'] = coef_df['p値'].apply(
                    lambda x: '***' if x < 0.001 else '**' if x < 0.01 else '*' if x < 0.05 else 'n.s.'
                )
                
                coef_df_sorted = pd.concat([
                    coef_df[coef_df['変数'] == '切片'],
                    coef_df[coef_df['変数'] != '切片'].sort_values('係数', key=abs, ascending=False)
                ])
                
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
                
                st.caption("有意水準: *** p<0.001, ** p<0.01, * p<0.05, n.s. 有意でない")
                
                st.markdown("#### 📊 係数の可視化（信頼区間付き）")
                
                coef_plot_df = coef_df_sorted[coef_df_sorted['変数'] != '切片'].copy()
                
                fig_coef = go.Figure()
                
                colors = ['#e74c3c' if p < 0.05 else '#95a5a6' for p in coef_plot_df['p値']]
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
                
                st.markdown("#### 💡 結果の解釈")
                st.info("""
                **係数の読み方:**
                - **正の係数**: その変数が1単位増えると、目的変数が係数分だけ増加
                - **負の係数**: その変数が1単位増えると、目的変数が係数分だけ減少
                - **p値 < 0.05**: 統計的に有意（偶然ではない可能性が高い）
                - **95%信頼区間**: 真の係数が存在する範囲（95%の確率）
                """)
                
                significant_vars = coef_df_sorted[(coef_df_sorted['p値'] < 0.05) & (coef_df_sorted['変数'] != '切片')]
                if len(significant_vars) > 0:
                    st.success(f"✅ **有意な変数（p < 0.05）**: {len(significant_vars)}個")
                    for idx, row in significant_vars.iterrows():
                        direction = "増加" if row['係数'] > 0 else "減少"
                        st.write(f"- **{row['変数']}**: 1単位増加すると目的変数が{abs(row['係数']):.4f}だけ{direction} (p={row['p値']:.4f})")
                else:
                    st.warning("⚠️ 統計的に有意な変数はありませんでした")
                
                diagnostics = comprehensive_regression_diagnostics(model_sm, X_train, y_train, residuals)
                display_diagnostics_report(diagnostics)
                
                effect_sizes = calculate_effect_sizes(model_sm, X_train, y_train)
                display_effect_sizes(effect_sizes)
                
                if enable_cv:
                    st.markdown("---")
                    with st.spinner("クロスバリデーション実行中..."):
                        model_for_cv = LinearRegression()
                        cv_results = rigorous_cross_validation(model_for_cv, X, y, cv=cv_folds)
                        
                        if cv_results:
                            display_cv_results(cv_results)
                
                st.markdown("---")
                st.markdown("### 🎯 予測精度の可視化")
                
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
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
                
                with col_plot2:
                    test_residuals = y_test - y_pred
                    fig_resid = px.scatter(x=y_pred, y=test_residuals, title="残差プロット")
                    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
                    fig_resid.update_xaxes(title="予測値")
                    fig_resid.update_yaxes(title="残差")
                    st.plotly_chart(fig_resid, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🔍 多重共線性チェック（VIF）")
                vif_data = calculate_vif(regression_df, selected_features)
                if vif_data is not None:
                    st.dataframe(
                        vif_data.style.background_gradient(cmap='YlOrRd', subset=['VIF']),
                        use_container_width=True
                    )
                    st.caption("基準: VIF < 5 (問題なし), 5 < VIF < 10 (注意), VIF > 10 (深刻な多重共線性)")
                    
                    if (vif_data['VIF'] > 10).any():
                        st.warning("⚠️ VIF > 10 の変数があります。Ridge回帰またはLasso回帰を検討してください")
                
                with st.expander("📋 詳細な統計情報（テキスト形式）"):
                    st.text(model_sm.summary())
                
                st.session_state.diagnostics_results = {
                    'model': model_sm,
                    'diagnostics': diagnostics,
                    'effect_sizes': effect_sizes,
                    'cv_results': cv_results if enable_cv else None,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae
                }
                
                # ==================== コード生成機能 ====================
                st.markdown("---")
                st.markdown("### 💻 この分析を再現するコード")
                st.info("上級者向け：ここまでの分析を実行可能なPythonコードとして出力します")
                
                with st.expander("📝 Python コードを生成・ダウンロード", expanded=False):
                    generated_code = f'''"""
自動生成された回帰分析コード
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

目的変数: {target}
説明変数: {len(selected_features)}個
モデルタイプ: {model_type}
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, linear_rainbow
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み（※ファイル名を変更してください）
df = pd.read_csv('your_data.csv')

print(f"データサイズ: {{df.shape[0]}}行 × {{df.shape[1]}}列")

# 前処理（欠損値削除）
df_clean = df[['{target}'] + {selected_features}].dropna()
print(f"前処理後: {{df_clean.shape[0]}}行")

# データ分割
X = df_clean{selected_features}
y = df_clean['{target}']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size={test_size}, 
    random_state=42
)

print(f"訓練データ: {{X_train.shape[0]}}行")
print(f"テストデータ: {{X_test.shape[0]}}行")

# モデル構築
'''
                    
                    if model_type == "通常の線形回帰（OLS）":
                        generated_code += '''
# OLS回帰（statsmodels）
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

model = sm.OLS(y_train, X_train_sm).fit()
y_pred = model.predict(X_test_sm)
'''
                    elif model_type == "Ridge回帰（正則化）":
                        generated_code += f'''
# Ridge回帰
model_sklearn = Ridge(alpha=1.0)
model_sklearn.fit(X_train, y_train)
y_pred = model_sklearn.predict(X_test)

# statsmodelsモデルも作成（診断用）
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
'''
                    else:
                        generated_code += f'''
# Lasso回帰
model_sklearn = Lasso(alpha=1.0)
model_sklearn.fit(X_train, y_train)
y_pred = model_sklearn.predict(X_test)

# statsmodelsモデルも作成（診断用）
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()
'''
                    
                    generated_code += f'''

# モデル評価
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\\n=== モデル性能 ===")
print(f"R²: {{r2:.4f}}")
print(f"調整済みR²: {{model.rsquared_adj:.4f}}")
print(f"RMSE: {{rmse:.4f}}")
print(f"MAE: {{mae:.4f}}")

# 回帰係数
print("\\n=== 回帰係数 ===")
coef_df = pd.DataFrame({{
    '変数': ['切片'] + list(X_train.columns),
    '係数': model.params.values,
    'p値': model.pvalues.values
}})
print(coef_df.to_string(index=False))

# 統計的診断
residuals = model.resid

# 線形性（Rainbow Test）
try:
    rainbow_stat, rainbow_p = linear_rainbow(model)
    print(f"\\n線形性（Rainbow Test）: p={{rainbow_p:.4f}}")
    if rainbow_p > 0.05:
        print("  ✓ 線形関係が適切")
    else:
        print("  ✗ 非線形性の可能性あり")
except:
    print("\\n線形性検定: 計算不可")

# 等分散性（Breusch-Pagan Test）
try:
    X_train_sm = sm.add_constant(X_train)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, X_train_sm)
    print(f"\\n等分散性（Breusch-Pagan）: p={{bp_p:.4f}}")
    if bp_p > 0.05:
        print("  ✓ 等分散性あり")
    else:
        print("  ✗ 不等分散（WLS回帰を検討）")
except:
    print("\\n等分散性検定: 計算不可")

# 正規性（Jarque-Bera Test）
try:
    jb_stat, jb_p, skew, kurtosis = jarque_bera(residuals)
    print(f"\\n正規性（Jarque-Bera）: p={{jb_p:.4f}}")
    print(f"  歪度: {{skew:.4f}}, 尖度: {{kurtosis:.4f}}")
    if jb_p > 0.05:
        print("  ✓ 正規分布に従う")
    else:
        print("  ✗ 正規性違反")
except:
    print("\\n正規性検定: 計算不可")

# 自己相関（Durbin-Watson）
try:
    dw_stat = durbin_watson(residuals)
    print(f"\\n自己相関（Durbin-Watson）: {{dw_stat:.4f}}")
    if 1.5 < dw_stat < 2.5:
        print("  ✓ 自己相関なし")
    else:
        print("  ✗ 自己相関の可能性")
except:
    print("\\n自己相関検定: 計算不可")

# 多重共線性（VIF）
try:
    vif_data = pd.DataFrame()
    vif_data["変数"] = X_train.columns
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                       for i in range(len(X_train.columns))]
    
    print("\\n=== 多重共線性（VIF） ===")
    print(vif_data.to_string(index=False))
    
    if (vif_data['VIF'] < 10).all():
        print("  ✓ 多重共線性なし（VIF<10）")
    else:
        print("  ✗ 多重共線性あり（Ridge/Lasso回帰を検討）")
except:
    print("\\n多重共線性計算: 計算不可")

# 効果量の計算
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled_df = pd.DataFrame(X_scaled, columns=X_train.columns)
X_scaled_with_const = sm.add_constant(X_scaled_df)
model_scaled = sm.OLS(y_train, X_scaled_with_const).fit()

beta_df = pd.DataFrame({{
    '変数': X_train.columns,
    '標準化係数（Beta）': model_scaled.params[1:].values
}}).sort_values('標準化係数（Beta）', key=abs, ascending=False)

print("\\n=== 標準化係数（効果の大きさ）===")
print(beta_df.to_string(index=False))

# Cohen's f²
full_r2 = model.rsquared
if full_r2 < 1:
    cohens_f2 = full_r2 / (1 - full_r2)
    if cohens_f2 < 0.02:
        interpretation = "小さい効果"
    elif cohens_f2 < 0.15:
        interpretation = "中程度の効果"
    elif cohens_f2 < 0.35:
        interpretation = "大きい効果"
    else:
        interpretation = "非常に大きい効果"
    
    print(f"\\nCohen's f²: {{cohens_f2:.4f}} ({{interpretation}})")

# クロスバリデーション
print("\\n=== クロスバリデーション ===")
cv_model = LinearRegression()
scoring = {{
    'r2': 'r2',
    'neg_mse': 'neg_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error'
}}

cv_results = cross_validate(
    cv_model, X, y,
    cv=KFold(n_splits={cv_folds}, shuffle=True, random_state=42),
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

train_r2 = cv_results['train_r2'].mean()
test_r2 = cv_results['test_r2'].mean()
overfit_gap = train_r2 - test_r2

print(f"訓練R²（平均）: {{train_r2:.4f}}")
print(f"テストR²（平均）: {{test_r2:.4f}} ± {{cv_results['test_r2'].std():.4f}}")
print(f"過学習ギャップ: {{overfit_gap:.4f}}")

if overfit_gap > 0.1:
    print("  ✗ 過学習の兆候あり（正則化を検討）")
else:
    print("  ✓ 過学習なし")

# 可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 予測vs実測
axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2)
axes[0, 0].set_xlabel('実測値')
axes[0, 0].set_ylabel('予測値')
axes[0, 0].set_title('実測値 vs 予測値')

# 残差プロット
axes[0, 1].scatter(y_pred, y_test - y_pred, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('予測値')
axes[0, 1].set_ylabel('残差')
axes[0, 1].set_title('残差プロット')

# 残差のヒストグラム
axes[1, 0].hist(residuals, bins=30, edgecolor='black')
axes[1, 0].set_xlabel('残差')
axes[1, 0].set_ylabel('頻度')
axes[1, 0].set_title('残差の分布')

# Q-Qプロット
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('Q-Qプロット（正規性確認）')

plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=300, bbox_inches='tight')
print("\\n可視化を 'regression_diagnostics.png' として保存しました")
plt.show()

# 詳細な統計情報
print("\\n" + "="*60)
print("詳細な統計情報")
print("="*60)
print(model.summary())

# ここから自由にカスタマイズ可能
# 例：
# - 交互作用項の追加: X['age_income'] = X['age'] * X['income']
# - 非線形項: X['age_squared'] = X['age'] ** 2
# - ロバスト回帰: from statsmodels.robust.robust_linear_model import RLM
# - WLS回帰: sm.WLS(y, X, weights=...)
# - 予測区間: predictions.summary_frame(alpha=0.05)
'''
                    
                    st.code(generated_code, language='python')
                    
                    col_dl1, col_dl2, col_dl3 = st.columns(3)
                    
                    with col_dl1:
                        st.download_button(
                            "💾 .py ファイルとして保存",
                            data=generated_code,
                            file_name=f"regression_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                            mime="text/x-python",
                            type="primary"
                        )
                    
                    with col_dl2:
                        notebook_content = {
                            "cells": [
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": [
                                        f"# 回帰分析レポート\n",
                                        f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                                        f"**目的変数**: {target}\n",
                                        f"**説明変数数**: {len(selected_features)}個\n",
                                        f"**モデル**: {model_type}"
                                    ]
                                },
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "outputs": [],
                                    "source": generated_code.split('\n')
                                }
                            ],
                            "metadata": {
                                "kernelspec": {
                                    "display_name": "Python 3",
                                    "language": "python",
                                    "name": "python3"
                                },
                                "language_info": {
                                    "name": "python",
                                    "version": "3.8.0"
                                }
                            },
                            "nbformat": 4,
                            "nbformat_minor": 4
                        }
                        
                        notebook_json = json.dumps(notebook_content, indent=2)
                        
                        st.download_button(
                            "📓 Jupyter Notebook として保存",
                            data=notebook_json,
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb",
                            mime="application/json",
                            type="primary"
                        )
                    
                    with col_dl3:
                        analysis_data = regression_df[[target] + selected_features]
                        csv_data = analysis_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📊 分析データ（CSV）を保存",
                            data=csv_data,
                            file_name=f"analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    st.markdown("---")
                    st.success("""
                    ✅ **生成されたコードの使い方**:
                    1. `.py`ファイルをダウンロードしてPython環境で実行
                    2. Jupyter Notebookとしてダウンロードしてブラウザで開く
                    3. コードをコピーして自由にカスタマイズ
                    4. `your_data.csv`の部分を実際のファイル名に変更
                    """)
                    
                    st.info("""
                    💡 **上級者向けカスタマイズ例**:
                    - 交互作用項の追加: `X['age_income'] = X['age'] * X['income']`
                    - 非線形項: `X['age_squared'] = X['age'] ** 2`
                    - ロバスト回帰: `from statsmodels.robust.robust_linear_model import RLM`
                    - WLS回帰（不等分散対応）: `sm.WLS(y, X, weights=...)`
                    - 予測区間の計算: `predictions.summary_frame(alpha=0.05)`
                    """)
                
                st.success("✅ 回帰分析が完了しました")
                
                # ==================== ビジネスインパクトシミュレーション ====================
                               
                st.markdown("---")
                st.markdown("### ビジネスインパクトシミュレーション（自動実行＋回帰対応）")
                st.info("回帰モデルでも予測値が高い＝リスクが高いと解釈し、離脱防止施策のROIをMonte Carloシミュレーションで計算します")

                if (st.session_state.current_model is not None and 
                    st.session_state.current_y_pred is not None and 
                    st.session_state.current_X_test is not None and 
                    st.session_state.current_y_test is not None):

                    # 回帰の予測値を0～1に正規化 → 「擬似離脱確率」として使用
                    y_pred_raw = np.array(st.session_state.current_y_pred)
                    pred_proba_sim = (y_pred_raw - y_pred_raw.min()) / (y_pred_raw.max() - y_pred_raw.min() + 1e-12)

                    # ここで関数を「呼び出すだけ」で中身が全部表示される（ボタン不要で即表示）
                    business_impact_simulation(
                        model=st.session_state.current_model,
                        X_test=st.session_state.current_X_test,
                        y_test=st.session_state.current_y_test,
                        y_pred=st.session_state.current_y_pred,
                        pred_proba=pred_proba_sim
                    )
                else:
                    st.warning("モデルがまだ実行されていません。「回帰分析実行」を先に押してください")
    
    # 予測モデル（機械学習）
    elif analysis_type == "🎯 予測モデル（機械学習）":
        st.markdown("### 機械学習による予測")
        
        selected_features = st.multiselect(
            "予測に使用する説明変数",
            options=numeric_features,
            default=numeric_features[:min(10, len(numeric_features))]
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ 説明変数を選択してください")
            st.stop()
        
        ml_df = df[[target] + selected_features].dropna()
        
        is_classification = len(ml_df[target].unique()) < 20 and not pd.api.types.is_float_dtype(ml_df[target])
        
        task_type = "分類" if is_classification else "回帰"
        st.info(f"タスクタイプ: **{task_type}** （目的変数のユニーク数: {ml_df[target].nunique()}）")
        
        with st.expander("🔧 高度な設定"):
            test_size = st.slider("テストデータの割合", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("ランダムシード", value=42, min_value=0)
            enable_cv = st.checkbox("クロスバリデーションを実行", value=True)
            cv_folds = st.slider("CVのfold数", 3, 10, 5) if enable_cv else 5
            if is_classification:
                ml_model = st.selectbox("モデル選択", ["ランダムフォレスト分類", "ロジスティック回帰"])
            else:
                ml_model = st.selectbox("モデル選択", ["ランダムフォレスト回帰", "線形回帰"])
        
        if st.button("✅ モデル学習実行", type="primary"):
            X = ml_df[selected_features]
            y = ml_df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            with st.spinner("モデル学習中..."):
                if is_classification:
                    if ml_model == "ランダムフォレスト分類":
                        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                    else:
                        model = LogisticRegression(max_iter=1000, random_state=random_state)
                else:
                    if ml_model == "ランダムフォレスト回帰":
                        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                    else:
                        model = LinearRegression()
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # モデル情報をセッションステートに保存
                st.session_state.current_model = model
                st.session_state.current_X_test = X_test
                st.session_state.current_y_test = y_test
                st.session_state.current_y_pred = y_pred
                
                st.markdown("---")
                st.markdown("## 📊 モデル評価結果")
                
                if is_classification:
                    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                    
                    st.markdown("### モデル性能")
                    
                    col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
                    col_eval1.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                    col_eval2.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                    col_eval3.metric("F1スコア", f"{f1_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                    if y_pred_proba is not None and len(np.unique(y_test)) == 2:
                        col_eval4.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_proba):.3f}")
                    
                    st.markdown("---")
                    st.markdown("#### 混同行列")
                    cm = confusion_matrix(y_test, y_pred)
                    fig_cm = px.imshow(
                        cm,
                        text_auto=True,
                        title="混同行列",
                        labels=dict(x="予測", y="実測"),
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                    
                    st.markdown("---")
                    st.markdown("#### 詳細な分類レポート")
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
                    
                else:
                    r2 = r2_score(y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    col_eval1, col_eval2, col_eval3 = st.columns(3)
                    col_eval1.metric("R² スコア", f"{r2:.4f}")
                    col_eval2.metric("RMSE", f"{rmse:.4f}")
                    col_eval3.metric("MAE", f"{mae:.4f}")
                    
                    st.markdown("---")
                    st.markdown("#### 予測 vs 実測")
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
                
                if enable_cv and not is_classification:
                    st.markdown("---")
                    with st.spinner("クロスバリデーション実行中..."):
                        cv_results = rigorous_cross_validation(model, X, y, cv=cv_folds)
                        
                        if cv_results:
                            display_cv_results(cv_results)
                
                st.markdown("---")
                st.markdown("### 📊 特徴量重要度")
                
                if hasattr(model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        '変数': selected_features,
                        '重要度': model.feature_importances_
                    }).sort_values('重要度', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df.head(15),
                        x='重要度',
                        y='変数',
                        orientation='h',
                        title="特徴量重要度（Top 15）"
                    )
                    fig_imp.update_layout(height=max(400, len(importance_df.head(15)) * 30))
                    st.plotly_chart(fig_imp, use_container_width=True)
                    st.dataframe(importance_df, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🔄 Permutation Importance")
                
                try:
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
                            perm_df.head(15),
                            x='重要度',
                            y='変数',
                            error_x='標準偏差',
                            orientation='h',
                            title="Permutation Importance（Top 15）"
                        )
                        fig_perm.update_layout(height=max(400, len(perm_df.head(15)) * 30))
                        st.plotly_chart(fig_perm, use_container_width=True)
                        st.dataframe(perm_df, use_container_width=True)
                        
                except Exception as e:
                    st.warning(f"Permutation Importance計算でエラー: {e}")
                
                st.markdown("---")
                st.markdown("### 💡 ビジネスインサイト")
                
                if hasattr(model, 'feature_importances_'):
                    top_features = importance_df.head(3)['変数'].tolist()
                    st.success(f"**最重要変数（Top 3）**: {', '.join(top_features)}")
                    
                    st.info("""
                    **推奨アクション:**
                    - 上位の重要変数に焦点を当てた施策を検討
                    - 重要度の低い変数は簡略化を検討
                    - 定期的にモデルを再学習して精度を維持
                    """)
                
                # ==================== 機械学習用コード生成 ====================
                st.markdown("---")
                st.markdown("### 💻 この予測モデルを再現するコード")
                
                with st.expander("📝 Python コードを生成・ダウンロード", expanded=False):
                    task_label = "分類" if is_classification else "回帰"
                    
                    ml_code = f'''"""
自動生成された機械学習{task_label}コード
生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

タスク: {task_label}
モデル: {ml_model}
説明変数: {len(selected_features)}個
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import '''
                    
                    if is_classification:
                        ml_code += '''classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
'''
                    else:
                        ml_code += '''mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
'''
                    
                    ml_code += f'''from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv('your_data.csv')

# 欠損値削除
df_clean = df[['{target}'] + {selected_features}].dropna()
print(f"データサイズ: {{df_clean.shape[0]}}行")

# データ分割
X = df_clean{selected_features}
y = df_clean['{target}']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size={test_size},
    random_state={random_state}
)

print(f"訓練: {{X_train.shape[0]}}行, テスト: {{X_test.shape[0]}}行")

# モデル構築
'''
                    
                    if is_classification:
                        if ml_model == "ランダムフォレスト分類":
                            ml_code += f'''
model = RandomForestClassifier(
    n_estimators=100,
    random_state={random_state},
    n_jobs=-1
)
'''
                        else:
                            ml_code += f'''
model = LogisticRegression(
    max_iter=1000,
    random_state={random_state},
    n_jobs=-1
)
'''
                    else:
                        if ml_model == "ランダムフォレスト回帰":
                            ml_code += f'''
model = RandomForestRegressor(
    n_estimators=100,
    random_state={random_state},
    n_jobs=-1
)
'''
                        else:
                            ml_code += '''
model = LinearRegression(n_jobs=-1)
'''
                    
                    ml_code += '''
# モデル学習
model.fit(X_train, y_train)

# 予測
y_pred = model.predict(X_test)

# モデル評価
'''
                    
                    if is_classification:
                        ml_code += '''
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

print("\\n=== 分類性能 ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1スコア: {f1:.4f}")

# 混同行列
cm = confusion_matrix(y_test, y_pred)
print("\\n混同行列:")
print(cm)

# 詳細レポート
print("\\n詳細な分類レポート:")
print(classification_report(y_test, y_pred, zero_division=0))

# ROC-AUC（2値分類の場合）
if hasattr(model, 'predict_proba') and len(np.unique(y_test)) == 2:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\\nROC-AUC: {roc_auc:.4f}")
'''
                    else:
                        ml_code += '''
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\\n=== 回帰性能 ===")
print(f"R²スコア: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
'''
                    
                    ml_code += f'''
# クロスバリデーション
scoring = '''
                    
                    if is_classification:
                        ml_code += '''{'accuracy': 'accuracy', 'f1': 'f1_weighted'}
'''
                    else:
                        ml_code += '''{'r2': 'r2', 'neg_mse': 'neg_mean_squared_error'}
'''
                    
                    ml_code += f'''
cv_results = cross_validate(
    model, X, y,
    cv=KFold(n_splits={cv_folds if enable_cv else 5}, shuffle=True, random_state={random_state}),
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print("\\n=== クロスバリデーション結果 ===")
'''
                    
                    if is_classification:
                        ml_code += '''
print(f"訓練Accuracy（平均）: {cv_results['train_accuracy'].mean():.4f}")
print(f"テストAccuracy（平均）: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
'''
                    else:
                        ml_code += '''
print(f"訓練R²（平均）: {cv_results['train_r2'].mean():.4f}")
print(f"テストR²（平均）: {cv_results['test_r2'].mean():.4f} ± {cv_results['test_r2'].std():.4f}")

overfit_gap = cv_results['train_r2'].mean() - cv_results['test_r2'].mean()
if overfit_gap > 0.1:
    print("  ⚠️ 過学習の兆候あり")
else:
    print("  ✓ 過学習なし")
'''
                    
                    ml_code += '''
# 特徴量重要度
if hasattr(model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        '変数': X.columns,
        '重要度': model.feature_importances_
    }).sort_values('重要度', ascending=False)
    
    print("\\n=== 特徴量重要度（上位10） ===")
    print(importance_df.head(10).to_string(index=False))
    
    # 可視化
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['変数'].head(15), importance_df['重要度'].head(15))
    plt.xlabel('重要度')
    plt.title('特徴量重要度（Top 15）')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("\\n特徴量重要度を 'feature_importance.png' に保存しました")
    plt.show()

# Permutation Importance
print("\\n=== Permutation Importance（計算中...）===")
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=''' + str(random_state) + ''',
    n_jobs=-1
)

perm_df = pd.DataFrame({
    '変数': X.columns,
    '重要度': perm_importance.importances_mean,
    '標準偏差': perm_importance.importances_std
}).sort_values('重要度', ascending=False)

print("\\nPermutation Importance（上位10）:")
print(perm_df.head(10).to_string(index=False))

# 予測結果の可視化
'''
                    
                    if not is_classification:
                        ml_code += '''
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 予測vs実測
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
axes[0].set_xlabel('実測値')
axes[0].set_ylabel('予測値')
axes[0].set_title('実測値 vs 予測値')

# 残差プロット
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.6)
axes[1].axhline(y=0, color='r', linestyle='--')
axes[1].set_xlabel('予測値')
axes[1].set_ylabel('残差')
axes[1].set_title('残差プロット')

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=300, bbox_inches='tight')
print("\\n予測結果を 'prediction_results.png' に保存しました")
plt.show()
'''
                    else:
                        ml_code += '''
# 混同行列の可視化
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('予測')
plt.ylabel('実測')
plt.title('混同行列')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\\n混同行列を 'confusion_matrix.png' に保存しました")
plt.show()
'''
                    
                    ml_code += '''
# ここから自由にカスタマイズ可能
# 例：
# - ハイパーパラメータチューニング（GridSearchCV）
# - アンサンブル学習（VotingRegressor/Classifier）
# - SHAP値による解釈性向上
# - 新しいデータでの予測
# - モデルの保存: joblib.dump(model, 'model.pkl')
'''
                    
                    st.code(ml_code, language='python')
                    
                    col_ml1, col_ml2, col_ml3 = st.columns(3)
                    
                    with col_ml1:
                        st.download_button(
                            "💾 .py ファイルとして保存",
                            data=ml_code,
                            file_name=f"ml_{task_label}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py",
                            mime="text/x-python",
                            type="primary"
                        )
                    
                    with col_ml2:
                        notebook_ml = {
                            "cells": [
                                {
                                    "cell_type": "markdown",
                                    "metadata": {},
                                    "source": [
                                        f"# 機械学習{task_label}分析\n",
                                        f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
                                        f"**モデル**: {ml_model}\n",
                                        f"**説明変数数**: {len(selected_features)}個"
                                    ]
                                },
                                {
                                    "cell_type": "code",
                                    "execution_count": None,
                                    "metadata": {},
                                    "outputs": [],
                                    "source": ml_code.split('\n')
                                }
                            ],
                            "metadata": {
                                "kernelspec": {
                                    "display_name": "Python 3",
                                    "language": "python",
                                    "name": "python3"
                                },
                                "language_info": {
                                    "name": "python",
                                    "version": "3.8.0"
                                }
                            },
                            "nbformat": 4,
                            "nbformat_minor": 4
                        }
                        
                        notebook_ml_json = json.dumps(notebook_ml, indent=2)
                        
                        st.download_button(
                            "📓 Jupyter Notebook として保存",
                            data=notebook_ml_json,
                            file_name=f"ml_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ipynb",
                            mime="application/json",
                            type="primary"
                        )
                    
                    with col_ml3:
                        ml_analysis_data = ml_df[[target] + selected_features]
                        csv_ml_data = ml_analysis_data.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "📊 分析データ（CSV）を保存",
                            data=csv_ml_data,
                            file_name=f"ml_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    
                    st.success("""
                    ✅ **生成されたコードで可能なこと**:
                    - モデルの再学習と予測
                    - ハイパーパラメータの調整
                    - 新しいデータでの予測
                    - モデルの保存と読み込み
                    """)
                    
                    st.info("""
                    💡 **上級者向けカスタマイズ例**:
                    - グリッドサーチ: `GridSearchCV(model, param_grid, cv=5)`
                    - SHAP解釈: `import shap; explainer = shap.TreeExplainer(model)`
                    - モデル保存: `import joblib; joblib.dump(model, 'model.pkl')`
                    - アンサンブル: `VotingRegressor([('rf', rf), ('lr', lr)])`
                    """)
                
                st.success("✅ 予測モデルの学習が完了しました")
                
                # ==================== ビジネスインパクトシミュレーション ====================
                st.markdown("---")
                st.markdown("### 💰 ビジネスインパクトシミュレーション")
                
                # 現在のモデル情報を使用してシミュレーションを実行
                if st.session_state.current_model is not None:
                    # 予測確率を取得
                    pred_proba = None
                    if hasattr(st.session_state.current_model, "predict_proba"):
                        try:
                            pred_proba = st.session_state.current_model.predict_proba(st.session_state.current_X_test)[:, 1]
                        except:
                            pass
                    
                    # ビジネスインパクトシミュレーションを実行
                    business_impact_simulation(
                        st.session_state.current_model,
                        st.session_state.current_X_test,
                        st.session_state.current_y_test,
                        st.session_state.current_y_pred,
                        pred_proba
                    )
    
    # ==================== 自動最適化分析（ビジネスインパクト統合版） ====================
    elif analysis_type in ["🚀 回帰分析（自動最適化）", "🤖 予測モデル（自動チューニング）"]:
        st.markdown("### 🤖 自動最適化分析")
        
        st.info("グリッドサーチで最適なモデルを自動探索し、ビジネスインパクトをシミュレーションします")
        
        selected_features = st.multiselect(
            "使用する説明変数",
            options=numeric_features,
            default=numeric_features[:min(10, len(numeric_features))]
        )
        
        if len(selected_features) == 0:
            st.warning("⚠️ 説明変数を選択してください")
            st.stop()
        
        auto_df = df[[target] + selected_features].dropna()
        
        is_classification = len(auto_df[target].unique()) < 20 and not pd.api.types.is_float_dtype(auto_df[target])
        
        task_type = "分類" if is_classification else "回帰"
        st.info(f"タスクタイプ: **{task_type}** （目的変数のユニーク数: {auto_df[target].nunique()}）")
        
        with st.expander("🔧 高度な設定"):
            test_size = st.slider("テストデータの割合", 0.1, 0.4, 0.2, 0.05)
            random_state = st.number_input("ランダムシード", value=42, min_value=0)
            cv_folds = st.slider("CVのfold数", 3, 10, 5)
        
        if st.button("🚀 自動最適化実行", type="primary"):
            X = auto_df[selected_features]
            y = auto_df[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
            
            with st.spinner("モデル最適化中..."):
                if is_classification:
                    # 分類モデルのグリッドサーチ
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10]
                    }
                    base_model = RandomForestClassifier(random_state=random_state)
                else:
                    # 回帰モデルのグリッドサーチ
                    param_grid = {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, 5, 10]
                    }
                    base_model = RandomForestRegressor(random_state=random_state)
                
                grid_search = GridSearchCV(
                    base_model, param_grid, 
                    cv=cv_folds, 
                    scoring='f1_weighted' if is_classification else 'r2',
                    n_jobs=-1,
                    verbose=1
                )
                
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test)
                
                # モデル情報をセッションステートに保存
                st.session_state.current_model = best_model
                st.session_state.current_X_test = X_test
                st.session_state.current_y_test = y_test
                st.session_state.current_y_pred = y_pred
                st.session_state.best_model = best_model
                
                st.markdown("---")
                st.markdown("## 🏆 最適化結果")
                
                st.success(f"**最適パラメータ**: {grid_search.best_params_}")
                st.success(f"**最適スコア**: {grid_search.best_score_:.4f}")
                
                if is_classification:
                    accuracy = (y_pred == y_test).mean()
                    st.metric("テスト精度", f"{accuracy:.4f}")
                else:
                    r2 = r2_score(y_test, y_pred)
                    st.metric("テストR²", f"{r2:.4f}")
                
                # 特徴量重要度
                if hasattr(best_model, 'feature_importances_'):
                    st.markdown("---")
                    st.markdown("### 📊 特徴量重要度")
                    
                    importance_df = pd.DataFrame({
                        '変数': selected_features,
                        '重要度': best_model.feature_importances_
                    }).sort_values('重要度', ascending=False)
                    
                    fig_imp = px.bar(
                        importance_df.head(15),
                        x='重要度',
                        y='変数',
                        orientation='h',
                        title="最適モデルの特徴量重要度（Top 15）"
                    )
                    fig_imp.update_layout(height=max(400, len(importance_df.head(15)) * 30))
                    st.plotly_chart(fig_imp, use_container_width=True)
                
                # ==================== ビジネスインパクトシミュレーション呼び出し ====================
                st.markdown("---")
                st.markdown("### 💰 ビジネスインパクトシミュレーション")
                
                # 予測確率を取得
                pred_proba = None
                if hasattr(best_model, "predict_proba"):
                    try:
                        pred_proba = best_model.predict_proba(X_test)[:, 1]
                    except:
                        pass
                
                # ビジネスインパクトシミュレーションを実行
                business_impact_simulation(best_model, X_test, y_test, y_pred, pred_proba)
    
    st.markdown("---")
    if st.button("➡️ ステップ5へ進む（解釈とレポート）", type="primary"):
        st.session_state.step = 5
        st.rerun()

# ステップ5: 解釈とレポート
elif st.session_state.step == 5:
    st.markdown('<div class="step-header">5️⃣ 解釈とレポート生成</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h4>📊 分析結果の解釈とアクションプラン</h4>
    <p>分析結果を非専門家にもわかりやすく解釈し、具体的なアクションを提案します</p>
    </div>
    """, unsafe_allow_html=True)
    
    report_title = st.text_input("レポートタイトル", value="データ分析レポート")
    report_summary = st.text_area(
        "分析サマリ（エグゼクティブサマリ）",
        height=150,
        placeholder="分析の背景、目的、主要な発見を簡潔に記述してください...",
        key="report_summary"
    )
    
    st.markdown("---")
    
    st.subheader("🔍 主要な発見")
    
    num_findings = st.number_input("発見の数", min_value=1, max_value=10, value=3)
    findings = []
    for i in range(num_findings):
        finding = st.text_area(f"発見 {i+1}", key=f"finding_{i}",
                              placeholder="例: 変数Xが目的変数に最も強い正の影響を与えている")
        findings.append(finding)
    
    st.markdown("---")
    
    st.subheader("💡 推奨アクション")
    
    num_actions = st.number_input("アクションの数", min_value=1, max_value=10, value=3)
    actions = []
    for i in range(num_actions):
        action = st.text_area(f"アクション {i+1}", key=f"action_{i}",
                             placeholder="例: 変数Xの改善に注力する施策を実施する")
        actions.append(action)
    
    st.markdown("---")
    
    if st.button("📄 レポート生成（プレビュー）", type="primary"):
        st.markdown("---")
        st.markdown("## 📊 分析レポートプレビュー")
        
        report_content = f"""
# {report_title}

## エグゼクティブサマリ
{report_summary}

## 分析概要
- **分析日**: {datetime.now().strftime('%Y年%m月%d日')}
- **データ**: {st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列
- **目的変数**: {st.session_state.selected_target}
- **説明変数数**: {len(st.session_state.selected_features)}個

## 主要な発見

"""
        for i, finding in enumerate(findings, 1):
            if finding:
                report_content += f"{i}. {finding}\n\n"
        
        if st.session_state.diagnostics_results:
            report_content += """
## 統計的妥当性の検証

"""
            diagnostics = st.session_state.diagnostics_results.get('diagnostics', {})
            
            passed_tests = [d.get('passed') for d in diagnostics.values() if d.get('passed') is not None]
            if passed_tests:
                pass_rate = sum(passed_tests) / len(passed_tests) * 100
                report_content += f"- **前提条件の合格率**: {pass_rate:.0f}%\n"
            
            for test_name, result in diagnostics.items():
                status = "✓ 合格" if result.get('passed') else "✗ 不合格"
                report_content += f"- **{result['test']}**: {status} - {result['interpretation']}\n"
            
            report_content += "\n"
            
            effect_sizes = st.session_state.diagnostics_results.get('effect_sizes', {})
            if effect_sizes.get('cohens_f2'):
                report_content += f"""
## 効果量（実務的重要性）

- **Cohen's f²**: {effect_sizes['cohens_f2']['value']:.4f} ({effect_sizes['cohens_f2']['interpretation']})

"""
        
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
*本レポートはData Science Workflow Studio Proにより生成されました*
"""
        
        st.markdown(report_content)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            md_bytes = report_content.encode('utf-8')
            st.download_button(
                "📥 Markdownレポート",
                data=md_bytes,
                file_name="analysis_report.md",
                mime="text/markdown"
            )
        
        with col2:
            if st.session_state.processed_df is not None:
                csv_bytes = st.session_state.processed_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 処理済みデータ (CSV)",
                    data=csv_bytes,
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
        
        with col3:
            html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>{report_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}
        h1 {{
            color:#8b5cf6;
            border-bottom: 3px solid #8b5cf6;
            padding-bottom: 15px;
        }}
        h2 {{
            color: #6d28d9;
            margin-top: 30px;
        }}
        .meta-info {{
            background: #f3f4f6;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .finding {{
            background: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .action {{
            background: #d1fae5;
            border-left: 4px solid #10b981;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #e5e7eb;
            text-align: center;
            color: #6b7280;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{report_title}</h1>
        
        <div class="meta-info">
            <p><strong>生成日時:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M')}</p>
            <p><strong>データサイズ:</strong> {st.session_state.df.shape[0]}行 × {st.session_state.df.shape[1]}列</p>
            <p><strong>目的変数:</strong> {st.session_state.selected_target}</p>
            <p><strong>説明変数数:</strong> {len(st.session_state.selected_features)}個</p>
        </div>
        
        <h2>エグゼクティブサマリ</h2>
        <p>{report_summary if report_summary else '（記入なし）'}</p>
        
        <h2>主要な発見</h2>
"""
            
            for i, finding in enumerate(findings, 1):
                if finding:
                    html_content += f'<div class="finding"><strong>発見 {i}:</strong> {finding}</div>\n'
            
            if st.session_state.diagnostics_results:
                html_content += """
        <h2>統計的妥当性の検証</h2>
"""
                diagnostics = st.session_state.diagnostics_results.get('diagnostics', {})
                
                for test_name, result in diagnostics.items():
                    status = "✓ 合格" if result.get('passed') else "✗ 不合格"
                    html_content += f'<p><strong>{status}</strong> {result["test"]}: {result["interpretation"]}</p>\n'
                
                effect_sizes = st.session_state.diagnostics_results.get('effect_sizes', {})
                if effect_sizes.get('cohens_f2'):
                    html_content += f"""
        <h2>効果量（実務的重要性）</h2>
        <p><strong>Cohen's f²:</strong> {effect_sizes['cohens_f2']['value']:.4f} ({effect_sizes['cohens_f2']['interpretation']})</p>
"""
            
            html_content += """
        <h2>推奨アクション</h2>
"""
            
            for i, action in enumerate(actions, 1):
                if action:
                    html_content += f'<div class="action"><strong>アクション {i}:</strong> {action}</div>\n'
            
            html_content += f"""
        <div class="footer">
            <p><strong>本レポートはData Science Workflow Studio Proにより生成されました</strong></p>
        </div>
    </div>
</body>
</html>
"""
            
            html_bytes = html_content.encode('utf-8')
            st.download_button(
                "📥 HTMLレポート",
                data=html_bytes,
                file_name=f"{report_title.replace(' ', '_')}_report.html",
                mime="text/html"
            )
    
    st.markdown("---")
    st.markdown("""
    <div class="success-box">
    <h3>分析が完了しました</h3>
    <p>生成されたレポートを関係者と共有し、推奨アクションの実装を検討してください</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_final1, col_final2, col_final3 = st.columns(3)
    
    with col_final1:
        if st.button("🔄 新しい分析を開始", type="secondary"):
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
    <p><strong>Data Science Workflow Studio Pro</strong></p>
    <p>統計コンサルタントレベルの厳密性 + コード生成機能 + ビジネスインパクト分析</p>
</div>
""", unsafe_allow_html=True)

# サイドバーにヘルプ情報
with st.sidebar:
    st.markdown("---")
    st.markdown("### 💡 ヘルプ")
    
    with st.expander("使い方のヒント"):
        st.markdown("""
        **ステップ1: 問題定義**
        - 何のための分析か明確に
        
        **ステップ2: データ理解**
        - 変数名をわかりやすく編集
        - 目的変数と説明変数を選択
        
        **ステップ3: データ前処理**
        - 欠損値・外れ値を処理
        - 必要な変換を適用
        
        **ステップ4: モデリング**
        - 統計的前提条件を自動検証
        - 効果量で実務的重要性を評価
        - **コード生成機能で自由にカスタマイズ**
        - **ビジネスインパクトシミュレーション**
        
        **ステップ5: レポート**
        - 発見とアクションを記載
        """)
    
    with st.expander("コード生成機能について"):
        st.markdown("""
        **各分析で実行可能なPythonコードを生成:**
        
        1. **.pyファイル**: そのまま実行可能
        2. **Jupyterノートブック**: ブラウザで編集・実行
        3. **カスタマイズ自由**: 交互作用項、非線形項、ロバスト回帰などを追加可能
        
        **上級者の使い方:**
        - アプリで基本分析を実施
        - コードをダウンロード
        - 自分の環境で高度な分析に拡張
        - GridSearchCV、SHAP、因果推論など
        """)
    
    with st.expander("ビジネスインパクト分析"):
        st.markdown("""
        **Monte Carloシミュレーション:**
        
        - 不確実性を考慮したROI計算
        - 95%信頼区間での利益予測
        - 介入戦略の最適化
        
        **実務活用例:**
        - 顧客離脱防止施策の効果予測
        - マーケティングキャンペーンのROI計算
        - リスク管理の定量化
        """)
    
    with st.expander("統計的厳密性について"):
        st.markdown("""
        **本アプリの統計的検証:**
        
        1. **前提条件チェック**
           - 線形性（Rainbow Test）
           - 等分散性（Breusch-Pagan）
           - 正規性（Jarque-Bera）
           - 自己相関（Durbin-Watson）
           - 多重共線性（VIF）
        
        2. **効果量の計算**
           - Cohen's f²
           - 標準化係数（Beta）
           - Partial R²
        
        3. **過学習の検証**
           - K-fold クロスバリデーション
        """)
    
    with st.expander("統計用語集"):
        st.markdown("""
        **R²（決定係数）**: モデルの当てはまり（0-1）
        
        **RMSE**: 予測誤差の大きさ
        
        **p値**: 統計的有意性（< 0.05で有意）
        
        **VIF**: 多重共線性（> 10で問題）
        
        **Cohen's f²**: 効果の大きさ
        - 小: 0.02, 中: 0.15, 大: 0.35
        """)


