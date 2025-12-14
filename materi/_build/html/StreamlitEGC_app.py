"""
ECG Time Series Classification App
Streamlit Application for ECGFiveDays Dataset Classification

Author: Data Science Team
Date: December 2024
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
import time
from scipy.signal import savgol_filter
import json

# Page configuration
st.set_page_config(
    page_title="ECG Time Series Classifier",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CACHING & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_models():
    """Load all trained models"""
    # --- KODE BARU (Ubah bagian ini) ---
    # Mengambil lokasi absolut di mana file .py ini berada
    current_dir = Path(__file__).parent
    
    # Menunjuk ke folder models yang ada di sebelah file .py
    models_dir = current_dir / 'models' 
    # -----------------------------------
    
    models = {}
    try:
        # Load KNN Euclidean (gunakan models_dir yang baru)
        with open(models_dir / 'knn_euclidean.pkl', 'rb') as f:
            models['1-NN Euclidean'] = pickle.load(f)
            
        # ... (kode selanjutnya sama, pastikan tetap pakai variable models_dir) ...

@st.cache_resource
def load_test_data():
    """Load test dataset for demo"""
    try:
        # --- KODE BARU (Ubah bagian ini) ---
        current_dir = Path(__file__).parent
        data_dir = current_dir / 'ECGFiveDays'
        # -----------------------------------
        
        data = np.loadtxt(data_dir / 'ECGFiveDays_TEST.txt')
        y = (data[:, 0] - 1).astype(int)  # Convert to [0, 1]
        X = data[:, 1:]
        return X, y
    except Exception as e:
        st.warning(f"Could not load test data: {e}")
        return None, None

@st.cache_data
def get_class_prototypes():
    """Calculate average signal per class"""
    X_test, y_test = load_test_data()
    if X_test is None:
        return None
    
    prototypes = {}
    for cls in [0, 1]:
        class_samples = X_test[y_test == cls]
        prototypes[cls] = np.mean(class_samples, axis=0)
    
    return prototypes

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_signal(signal, apply_winsorization=True, apply_normalization=True, 
                     apply_denoising=True):
    """
    Apply preprocessing pipeline to signal
    
    Parameters:
    -----------
    signal : numpy array (n_timepoints,)
    apply_winsorization : bool
    apply_normalization : bool
    apply_denoising : bool
    
    Returns:
    --------
    processed_signal : numpy array
    steps_applied : list of str
    """
    processed = signal.copy()
    steps = []
    
    # Winsorization
    if apply_winsorization:
        lb = np.percentile(processed, 1)
        ub = np.percentile(processed, 99)
        processed = np.clip(processed, lb, ub)
        steps.append("Winsorization (1-99 percentile)")
    
    # Z-score normalization
    if apply_normalization:
        mean = np.mean(processed)
        std = np.std(processed)
        if std > 1e-8:
            processed = (processed - mean) / std
            steps.append("Z-score Normalization")
        else:
            processed = processed - mean
            steps.append("Mean Centering (std~0)")
    
    # Denoising
    if apply_denoising:
        processed = savgol_filter(processed, window_length=11, polyorder=3)
        steps.append("Savitzky-Golay Smoothing")
    
    return processed, steps

def extract_features(signal):
    """Extract features for feature-based models"""
    from scipy.stats import skew, kurtosis
    
    # --- FITUR DASAR (9 Fitur) ---
    features = [
        np.mean(signal),
        np.std(signal),
        np.min(signal),
        np.max(signal),
        np.median(signal),
        skew(signal),
        kurtosis(signal),
        np.max(signal) - np.min(signal),  # range
        np.var(signal),
    ]
    # Hitung turunan pertama
    diff1 = np.diff(signal)

    # Tambahkan 1 fitur lagi: Rata-rata dari turunan pertama
    features.append(np.mean(diff1))
    
    # # 1st derivative
    # diff1 = np.diff(signal)
    # features.extend([
    #     np.mean(diff1),
    #     np.std(diff1),
    #     np.min(diff1),
    #     np.max(diff1),
    # ])
    
    # # 2nd derivative
    # diff2 = np.diff(diff1)
    # features.extend([
    #     np.mean(diff2),
    #     np.std(diff2),
    # ])
    
    return np.array(features).reshape(1, -1)

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_with_model(model, signal, model_name):
    """Make prediction with a single model"""
    start_time = time.time()
    
    if model_name == 'XGBoost':
        # Feature-based model
        features = extract_features(signal)
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
    else:
        # Distance-based model (1-NN)
        signal_2d = signal.reshape(1, -1)
        prediction = model.predict(signal_2d)[0]
        proba = model.predict_proba(signal_2d)[0]
    
    inference_time = time.time() - start_time
    
    return {
        'prediction': int(prediction),
        'probability': proba,
        'confidence': float(np.max(proba)),
        'inference_time': inference_time
    }

def ensemble_predict(models, signal):
    """Ensemble prediction using voting"""
    predictions = []
    probabilities = []
    
    for model_name, model in models.items():
        result = predict_with_model(model, signal, model_name)
        predictions.append(result['prediction'])
        probabilities.append(result['probability'])
    
    # Majority voting
    ensemble_pred = int(np.round(np.mean(predictions)))
    
    # Average probabilities
    ensemble_proba = np.mean(probabilities, axis=0)
    
    return {
        'prediction': ensemble_pred,
        'probability': ensemble_proba,
        'confidence': float(np.max(ensemble_proba)),
        'individual_predictions': predictions
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_signal(signal, title="Signal", color='steelblue', ax=None):
    """Plot time series signal"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    ax.plot(signal, linewidth=2, color=color)
    ax.set_xlabel('Time Points', fontweight='bold')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    return ax

def plot_comparison(raw_signal, processed_signal):
    """Plot raw vs processed signal"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Raw signal
    axes[0].plot(raw_signal, linewidth=2, color='coral', alpha=0.8)
    axes[0].set_xlabel('Time Points', fontweight='bold')
    axes[0].set_ylabel('Amplitude', fontweight='bold')
    axes[0].set_title('Raw Signal', fontweight='bold', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Processed signal
    axes[1].plot(processed_signal, linewidth=2, color='steelblue')
    axes[1].set_xlabel('Time Points', fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontweight='bold')
    axes[1].set_title('Preprocessed Signal', fontweight='bold', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_prediction_confidence(probabilities, class_names=['Class 0', 'Class 1']):
    """Plot prediction confidence as horizontal bar"""
    fig, ax = plt.subplots(figsize=(10, 3))
    
    colors = ['#ff7f0e', '#1f77b4']
    y_pos = np.arange(len(class_names))
    
    bars = ax.barh(y_pos, probabilities * 100, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Confidence (%)', fontweight='bold')
    ax.set_title('Prediction Confidence Distribution', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Annotate bars
    for bar, prob in zip(bars, probabilities):
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                f'{prob*100:.1f}%', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_model_comparison(results_dict):
    """Compare predictions from multiple models"""
    model_names = list(results_dict.keys())
    confidences = [results_dict[name]['confidence'] * 100 for name in model_names]
    predictions = [results_dict[name]['prediction'] for name in model_names]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if p == predictions[0] else 'orange' for p in predictions]
    bars = ax.barh(model_names, confidences, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Confidence (%)', fontweight='bold', fontsize=12)
    ax.set_title('Model Predictions Comparison', fontweight='bold', fontsize=14)
    ax.set_xlim(0, 100)
    ax.grid(axis='x', alpha=0.3)
    
    # Annotate
    for i, (bar, conf, pred) in enumerate(zip(bars, confidences, predictions)):
        ax.text(conf + 1, bar.get_y() + bar.get_height()/2,
                f'Class {pred} ({conf:.1f}%)', 
                ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_prototype_comparison(signal, prototypes):
    """Compare signal with class prototypes"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(signal, linewidth=2, label='Input Signal', color='black', alpha=0.8)
    ax.plot(prototypes[0], linewidth=2, label='Class 0 Prototype', 
            color='#ff7f0e', linestyle='--', alpha=0.7)
    ax.plot(prototypes[1], linewidth=2, label='Class 1 Prototype', 
            color='#1f77b4', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Time Points', fontweight='bold')
    ax.set_ylabel('Amplitude', fontweight='bold')
    ax.set_title('Signal Comparison with Class Prototypes', fontweight='bold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown('<p class="main-header">🫀 ECG Time Series Classifier</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Binary Classification using Machine Learning</p>', 
                unsafe_allow_html=True)
    
    # Load models
    models, metadata = load_models()
    X_test, y_test = load_test_data()
    prototypes = get_class_prototypes()
    
    if models is None:
        st.error("⚠️ Models not found! Please ensure models are in 'models/' directory.")
        st.stop()
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Input method selection
    input_method = st.sidebar.radio(
        "Select Input Method:",
        ["📤 Upload File", "🔢 Select from Test Dataset", "🎯 Use Demo Data"]
    )
    
    signal_raw = None
    signal_source = ""
    
    # ========================================================================
    # INPUT HANDLING
    # ========================================================================
    
    if input_method == "📤 Upload File":
        st.sidebar.markdown("---")
        uploaded_file = st.sidebar.file_uploader(
            "Upload ECG Signal File",
            type=['txt', 'csv', 'npy'],
            help="Upload a file containing time series data (136 time points)"
        )
        
        if uploaded_file is not None:
            try:
                # 1. BACA FILE
                if uploaded_file.name.endswith('.txt'):
                    signal_raw = np.loadtxt(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    signal_raw = pd.read_csv(uploaded_file, header=None).values.flatten()
                elif uploaded_file.name.endswith('.npy'):
                    signal_raw = np.load(uploaded_file)
                
                # 2. RATAKAN ARRAY (FLATTEN)
                if signal_raw.ndim > 1:
                    if signal_raw.shape[0] == 1 or signal_raw.shape[1] == 1:
                        signal_raw = signal_raw.flatten()
                    else:
                        # Jika ada banyak kolom/baris, ambil yang relevan
                        signal_raw = signal_raw.flatten()

                # 3. PENGECEKAN & PEMOTONGAN DATA (PENTING!)
                REQUIRED_LENGTH = 136
                current_len = len(signal_raw)

                if current_len != REQUIRED_LENGTH:
                    if current_len > REQUIRED_LENGTH:
                        # KASUS KAMU: Data 117820 dipotong jadi 136
                        st.sidebar.warning(f"⚠️ Data terlalu panjang ({current_len}). Memotong ke {REQUIRED_LENGTH} titik pertama.")
                        signal_raw = signal_raw[:REQUIRED_LENGTH]
                    else:
                        st.sidebar.error(f"❌ Data terlalu pendek ({current_len}). Wajib {REQUIRED_LENGTH} titik.")
                        signal_raw = None # Batalkan proses jika kependekan
                
                # 4. SUKSES
                if signal_raw is not None:
                    signal_source = f"Uploaded: {uploaded_file.name}"
                    st.sidebar.success(f"✅ File loaded: {len(signal_raw)} time points")
                
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {e}")
    
    elif input_method == "🔢 Select from Test Dataset":
        if X_test is not None:
            st.sidebar.markdown("---")
            
            # Class filter
            class_filter = st.sidebar.radio("Filter by class:", ["All", "Class 0", "Class 1"])
            
            if class_filter == "All":
                available_indices = list(range(len(X_test)))
            else:
                target_class = 0 if class_filter == "Class 0" else 1
                available_indices = np.where(y_test == target_class)[0].tolist()
            
            # Sample selection
            sample_idx = st.sidebar.selectbox(
                "Select sample index:",
                available_indices,
                format_func=lambda x: f"Sample {x} (Class {y_test[x]})"
            )
            
            signal_raw = X_test[sample_idx]
            true_label = y_test[sample_idx]
            signal_source = f"Test Dataset Sample #{sample_idx} (True Label: Class {true_label})"
            
            st.sidebar.success(f"✅ Sample selected")
    
    else:  # Demo Data
        st.sidebar.markdown("---")
        demo_choice = st.sidebar.selectbox(
            "Select demo signal:",
            ["Demo 1: Class 0 Example", "Demo 2: Class 1 Example", "Demo 3: Ambiguous Case"]
        )
        
        if X_test is not None:
            if "Demo 1" in demo_choice:
                signal_raw = X_test[y_test == 0][0]
                signal_source = "Demo: Typical Class 0 Signal"
            elif "Demo 2" in demo_choice:
                signal_raw = X_test[y_test == 1][0]
                signal_source = "Demo: Typical Class 1 Signal"
            else:
                # Find ambiguous sample (close to decision boundary)
                signal_raw = X_test[len(X_test)//2]
                signal_source = "Demo: Ambiguous Signal"
        else:
            # Fallback synthetic data
            signal_raw = np.sin(np.linspace(0, 10, 136)) + np.random.normal(0, 0.1, 136)
            signal_source = "Demo: Synthetic Signal"
        
        st.sidebar.success("✅ Demo data loaded")
    
    # Preprocessing options
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 Preprocessing")
    preprocessing_enabled = st.sidebar.checkbox("Apply Preprocessing", value=True)
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("🤖 Model Selection")
    available_models = list(models.keys())
    available_models.append("🎯 Ensemble (All Models)")
    
    selected_model = st.sidebar.selectbox(
        "Choose model:",
        available_models
    )
    
    # ========================================================================
    # MAIN CONTENT - TABS
    # ========================================================================
    
    if signal_raw is not None:
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Analysis", "🎯 Prediction", "📈 Comparison", "ℹ️ About"])
        
        # ====================================================================
        # TAB 1: ANALYSIS
        # ====================================================================
        with tab1:
            st.header("📊 Signal Analysis")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Signal Information")
                st.info(f"**Source:** {signal_source}")
                
                # Signal statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                with stats_col1:
                    st.metric("Length", len(signal_raw))
                with stats_col2:
                    st.metric("Mean", f"{np.mean(signal_raw):.3f}")
                with stats_col3:
                    st.metric("Std Dev", f"{np.std(signal_raw):.3f}")
                with stats_col4:
                    st.metric("Range", f"{np.ptp(signal_raw):.3f}")
            
            with col2:
                st.subheader("Quick Stats")
                st.write(f"**Min:** {np.min(signal_raw):.3f}")
                st.write(f"**Max:** {np.max(signal_raw):.3f}")
                st.write(f"**Median:** {np.median(signal_raw):.3f}")
            
            # Visualizations
            st.subheader("Signal Visualization")
            
            if preprocessing_enabled:
                # Preprocess
                signal_processed, steps = preprocess_signal(signal_raw)
                
                # Show comparison
                fig_comp = plot_comparison(signal_raw, signal_processed)
                st.pyplot(fig_comp)
                
                st.success("✅ **Preprocessing Steps Applied:**")
                for step in steps:
                    st.write(f"  • {step}")
            
            else:
                # Show raw signal only
                fig, ax = plt.subplots(figsize=(12, 5))
                plot_signal(signal_raw, "Raw Signal", ax=ax)
                st.pyplot(fig)
                
                st.warning("⚠️ Preprocessing disabled - using raw signal")
            
            # Prototype comparison
            if prototypes is not None:
                st.subheader("Comparison with Class Prototypes")
                signal_to_compare = signal_processed if preprocessing_enabled else signal_raw
                fig_proto = plot_prototype_comparison(signal_to_compare, prototypes)
                st.pyplot(fig_proto)
        
        # ====================================================================
        # TAB 2: PREDICTION
        # ====================================================================
        with tab2:
            st.header("🎯 Prediction Results")
            
            # Preprocess if enabled
            if preprocessing_enabled:
                signal_for_prediction, _ = preprocess_signal(signal_raw)
            else:
                signal_for_prediction = signal_raw
            
            # Make prediction
            with st.spinner("Making prediction..."):
                if "Ensemble" in selected_model:
                    # Ensemble prediction
                    result = ensemble_predict(models, signal_for_prediction)
                    is_ensemble = True
                else:
                    # Single model prediction
                    model = models[selected_model]
                    result = predict_with_model(model, signal_for_prediction, selected_model)
                    is_ensemble = False
            
            # Display results
            st.markdown("---")
            
            # Main prediction result
            pred_class = result['prediction']
            confidence = result['confidence']
            
            # Success box with result
            result_color = "success" if confidence > 0.8 else "warning" if confidence > 0.6 else "error"
            
            st.markdown(f"""
            <div class="{result_color}-box">
                <h2 style="margin:0;">🎯 Predicted Class: {pred_class}</h2>
                <h3 style="margin:0.5rem 0 0 0;">Confidence: {confidence*100:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Class", pred_class)
            with col2:
                st.metric("Confidence", f"{confidence*100:.1f}%")
            with col3:
                if 'inference_time' in result:
                    st.metric("Inference Time", f"{result['inference_time']*1000:.1f} ms")
            
            st.markdown("---")
            
            # Probability distribution
            st.subheader("Confidence Distribution")
            fig_conf = plot_prediction_confidence(result['probability'])
            st.pyplot(fig_conf)
            
            # Model info
            st.subheader("Model Information")
            if is_ensemble:
                st.info(f"**Model:** Ensemble (Voting from {len(models)} models)")
                st.write("Individual predictions:")
                for i, (model_name, pred) in enumerate(zip(models.keys(), result['individual_predictions'])):
                    st.write(f"  • {model_name}: Class {pred}")
            else:
                st.info(f"**Model:** {selected_model}")
            
            # Download results
            st.markdown("---")
            st.subheader("📥 Download Results")
            
            result_dict = {
                'predicted_class': int(pred_class),
                'confidence': float(confidence),
                'model': selected_model,
                'preprocessing': preprocessing_enabled,
                'probabilities': {
                    'class_0': float(result['probability'][0]),
                    'class_1': float(result['probability'][1])
                }
            }
            
            result_json = json.dumps(result_dict, indent=2)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Download Prediction (JSON)",
                    data=result_json,
                    file_name="prediction_result.json",
                    mime="application/json"
                )
            
            with col2:
                # Download preprocessed signal
                if preprocessing_enabled:
                    signal_csv = pd.DataFrame({'timepoint': range(len(signal_for_prediction)),
                                              'amplitude': signal_for_prediction})
                    st.download_button(
                        label="Download Preprocessed Signal (CSV)",
                        data=signal_csv.to_csv(index=False),
                        file_name="preprocessed_signal.csv",
                        mime="text/csv"
                    )
        
        # ====================================================================
        # TAB 3: COMPARISON
        # ====================================================================
        with tab3:
            st.header("📈 Model Comparison")
            
            st.info("Comparing predictions from all available models...")
            
            # Preprocess signal
            if preprocessing_enabled:
                signal_for_prediction, _ = preprocess_signal(signal_raw)
            else:
                signal_for_prediction = signal_raw
            
            # Get predictions from all models
            all_results = {}
            
            with st.spinner("Running all models..."):
                for model_name, model in models.items():
                    result = predict_with_model(model, signal_for_prediction, model_name)
                    all_results[model_name] = result
                
                # Add ensemble
                ensemble_result = ensemble_predict(models, signal_for_prediction)
                all_results['Ensemble'] = ensemble_result
            
            # Comparison visualization
            fig_comparison = plot_model_comparison(all_results)
            st.pyplot(fig_comparison)
            
            # Detailed comparison table
            st.subheader("Detailed Comparison")
            
            comparison_data = []
            for model_name, result in all_results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Prediction': result['prediction'],
                    'Class 0 Prob': f"{result['probability'][0]*100:.2f}%",
                    'Class 1 Prob': f"{result['probability'][1]*100:.2f}%",
                    'Confidence': f"{result['confidence']*100:.2f}%",
                    'Inference Time (ms)': f"{result.get('inference_time', 0)*1000:.2f}" if 'inference_time' in result else 'N/A'
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
            
            # Consensus analysis
            st.subheader("Consensus Analysis")
            predictions = [r['prediction'] for r in all_results.values()]
            unique, counts = np.unique(predictions, return_counts=True)
            
            if len(unique) == 1:
                st.success(f"✅ **Perfect consensus!** All models predict Class {unique[0]}")
            else:
                st.warning(f"⚠️ **Split decision:** Class 0: {counts[0]}, Class 1: {counts[1]}")
        
        # ====================================================================
        # TAB 4: ABOUT
        # ====================================================================
        with tab4:
            st.header("ℹ️ About This Application")
            
            st.markdown("""
            ### 🫀 ECG Time Series Classification
            
            This application performs binary classification on ECG (Electrocardiogram) time series data
            using machine learning models trained on the ECGFiveDays dataset.
            
            #### 📊 Dataset Information
            - **Name:** ECGFiveDays (UCR Time Series Archive)
            - **Type:** Binary Classification
            - **Time Points:** 136 per signal
            - **Classes:** 2 (Class 0 and Class 1)
            
            #### 🤖 Models Available
            """)
            
            for model_name in models.keys():
                with st.expander(f"📌 {model_name}"):
                    if model_name == "1-NN Euclidean":
                        st.write("""
                        **1-Nearest Neighbor with Euclidean Distance**
                        - Simple and fast baseline model
                        - Measures straight-line distance between signals
                        - Good for similar-shaped signals
                        """)
                    elif model_name == "XGBoost":
                        st.write("""
                        **XGBoost Classifier**
                        - Gradient boosting decision trees
                        - Works on extracted features (15 features)
                        - Often best performer for tabular/feature data
                        - Features: statistical, derivatives, temporal
                        """)
            
            with st.expander("📌 Ensemble Model"):
                st.write("""
                **Ensemble (Voting)**
                - Combines predictions from all models
                - Uses majority voting for final prediction
                - Averages probabilities for confidence
                - Generally more robust than individual models
                """)
            
            st.markdown("""
            #### 🔧 Preprocessing Pipeline
            
            When enabled, the following steps are applied:
            1. **Winsorization:** Clip extreme values (1-99 percentile)
            2. **Z-score Normalization:** Zero mean, unit variance
            3. **Savitzky-Golay Smoothing:** Noise reduction while preserving shape
            
            #### 📈 Model Performance
            """)
            
            if metadata:
                st.write(f"**Best Model:** {metadata.get('best_model', 'N/A')}")
                st.write(f"**Best Accuracy:** {metadata.get('best_accuracy', 0)*100:.2f}%")
                
                if 'results' in metadata:
                    results_df = pd.DataFrame(metadata['results'])
                    st.dataframe(results_df[['Model', 'Accuracy', 'F1-Score', 'ROC-AUC']], use_container_width=True)
            
            st.markdown("""
            #### 📚 How to Use
            
            1. **Select Input Method:**
               - Upload your own ECG signal file
               - Choose from test dataset samples
               - Use demo data for quick testing
            
            2. **Configure Settings:**
               - Toggle preprocessing on/off
               - Select model for prediction
            
            3. **Analyze Results:**
               - View signal visualization
               - Check prediction with confidence
               - Compare multiple models
            
            4. **Download Results:**
               - Export predictions as JSON
               - Save preprocessed signals as CSV
            
            #### 🎯 Expected Input Format
            
            Your uploaded file should contain:
            - **136 time points** (signal values)
            - Formats supported: `.txt`, `.csv`, `.npy`
            - One signal per file
            - Example format:
              ```
              -0.123
              0.456
              0.789
              ...
              ```
            
            #### 👨‍💻 Technical Details
            
            - **Framework:** Streamlit
            - **ML Libraries:** scikit-learn, XGBoost
            - **Preprocessing:** SciPy
            - **Visualization:** Matplotlib, Seaborn
            
            #### 📖 References
            
            - [UCR Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)
            - [ECGFiveDays Dataset](https://www.timeseriesclassification.com/description.php?Dataset=ECGFiveDays)
            
            #### 📝 Project Information
            
            - **Course:** Data Science Project
            - **Dataset:** ECGFiveDays
            - **Methodology:** CRISP-DM
            - **Date:** December 2024
            
            ---
            
            Made with ❤️ using Streamlit
            """)
    
    else:
        # No signal loaded yet
        st.info("👈 Please select an input method from the sidebar to begin")
        
        # Show quick start guide
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📤 Upload File
            Upload your own ECG signal file (.txt, .csv, .npy)
            - 136 time points required
            - One signal per file
            """)
        
        with col2:
            st.markdown("""
            ### 🔢 Test Dataset
            Select from 861 test samples
            - Real ECG signals
            - Known ground truth
            - Filter by class
            """)
        
        with col3:
            st.markdown("""
            ### 🎯 Demo Data
            Quick demonstration
            - Pre-loaded examples
            - Typical patterns
            - Ready to predict
            """)
        
        # Show example signals
        if X_test is not None:
            st.markdown("---")
            st.subheader("Example Signals from Dataset")
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 4))
            
            # Example from each class
            for i, cls in enumerate([0, 1]):
                sample = X_test[y_test == cls][0]
                axes[i].plot(sample, linewidth=2, color=f'C{i}')
                axes[i].set_title(f'Example: Class {cls}', fontweight='bold', fontsize=12)
                axes[i].set_xlabel('Time Points')
                axes[i].set_ylabel('Amplitude')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()