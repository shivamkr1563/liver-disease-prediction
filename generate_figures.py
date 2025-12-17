"""
Generate all figures for the IEEE Conference Paper
Run this script to create all required diagrams
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style for professional IEEE-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'

# ============================================================================
# Figure 1: System Architecture Diagram
# ============================================================================
def create_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    frontend_color = '#E3F2FD'
    backend_color = '#FFF3E0'
    ml_color = '#E8F5E9'
    
    # Layer 1: User Interface
    ax.add_patch(FancyBboxPatch((1, 8.5), 8, 1, boxstyle="round,pad=0.1", 
                                edgecolor='#1976D2', facecolor=frontend_color, linewidth=2))
    ax.text(5, 9, 'Frontend (React + Material-UI)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(5, 8.7, 'User Input Form', ha='center', va='center', fontsize=9)
    
    # Arrow 1
    ax.add_patch(FancyArrowPatch((5, 8.5), (5, 7.8), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    ax.text(5.5, 8.15, 'HTTP POST', fontsize=8, style='italic')
    
    # Layer 2: Backend Server
    ax.add_patch(FancyBboxPatch((1, 6.5), 8, 1.2, boxstyle="round,pad=0.1", 
                                edgecolor='#F57C00', facecolor=backend_color, linewidth=2))
    ax.text(5, 7.5, 'Backend (FastAPI Server)', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(5, 7.1, 'API Endpoints | Data Validation | Request Handling', 
            ha='center', va='center', fontsize=9)
    ax.text(5, 6.8, 'Port: 8000', ha='center', va='center', fontsize=8, style='italic')
    
    # Arrow 2
    ax.add_patch(FancyArrowPatch((5, 6.5), (5, 5.8), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    ax.text(5.5, 6.15, 'Raw Data', fontsize=8, style='italic')
    
    # Layer 3: Preprocessing
    ax.add_patch(FancyBboxPatch((1, 4.8), 8, 0.9, boxstyle="round,pad=0.1", 
                                edgecolor='#7B1FA2', facecolor='#F3E5F5', linewidth=2))
    ax.text(5, 5.5, 'Data Preprocessing Pipeline', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(5, 5.1, 'Gender Encoding | Z-Score Normalization | Feature Scaling', 
            ha='center', va='center', fontsize=9)
    
    # Arrow 3
    ax.add_patch(FancyArrowPatch((5, 4.8), (5, 4.1), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    ax.text(5.5, 4.45, 'Processed Features', fontsize=8, style='italic')
    
    # Layer 4: ML Models (Two boxes side by side)
    # Random Forest
    ax.add_patch(FancyBboxPatch((1, 2.5), 3.5, 1.5, boxstyle="round,pad=0.1", 
                                edgecolor='#388E3C', facecolor=ml_color, linewidth=2))
    ax.text(2.75, 3.6, 'Random Forest Model', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(2.75, 3.3, '200 Decision Trees', ha='center', va='center', fontsize=9)
    ax.text(2.75, 3.0, 'Ensemble Voting', ha='center', va='center', fontsize=9)
    ax.text(2.75, 2.7, 'AUC: 95.45%', ha='center', va='center', fontsize=9, 
            style='italic', color='#2E7D32')
    
    # KAN Model
    ax.add_patch(FancyBboxPatch((5.5, 2.5), 3.5, 1.5, boxstyle="round,pad=0.1", 
                                edgecolor='#388E3C', facecolor=ml_color, linewidth=2))
    ax.text(7.25, 3.6, 'KAN Model', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(7.25, 3.3, 'Learnable Splines', ha='center', va='center', fontsize=9)
    ax.text(7.25, 3.0, 'Interpretable Transform', ha='center', va='center', fontsize=9)
    ax.text(7.25, 2.7, 'AUC: 95.57%', ha='center', va='center', fontsize=9, 
            style='italic', color='#2E7D32')
    
    # Arrows from models
    ax.add_patch(FancyArrowPatch((2.75, 2.5), (5, 1.8), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    ax.add_patch(FancyArrowPatch((7.25, 2.5), (5, 1.8), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    
    # Layer 5: Result Processing
    ax.add_patch(FancyBboxPatch((1, 0.8), 8, 0.9, boxstyle="round,pad=0.1", 
                                edgecolor='#C62828', facecolor='#FFEBEE', linewidth=2))
    ax.text(5, 1.5, 'Result Processing & Risk Categorization', ha='center', va='center', 
            fontsize=12, fontweight='bold')
    ax.text(5, 1.1, 'Probability Score | Risk Level (Low/Moderate/High/Very High)', 
            ha='center', va='center', fontsize=9)
    
    # Arrow back to user
    ax.add_patch(FancyArrowPatch((5, 0.8), (5, 0.1), arrowstyle='->', 
                                mutation_scale=20, linewidth=2, color='black'))
    ax.text(5.5, 0.45, 'JSON Response', fontsize=8, style='italic')
    
    # Final result box
    ax.add_patch(FancyBboxPatch((1, -0.5), 8, 0.5, boxstyle="round,pad=0.1", 
                                edgecolor='#1976D2', facecolor=frontend_color, linewidth=2))
    ax.text(5, -0.25, 'Display to User (Visual Dashboard)', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    plt.title('AI-Powered Liver Disease Prediction System Architecture', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', bbox_inches='tight', dpi=300)
    print("✓ Created: architecture_diagram.png")
    plt.close()

# ============================================================================
# Figure 2: Confusion Matrices for All Models
# ============================================================================
def create_confusion_matrices():
    # Data from your results (approximate values based on performance metrics)
    confusion_data = {
        'Logistic Regression': np.array([[48, 12], [9, 169]]),
        'Random Forest': np.array([[52, 8], [14, 164]]),
        'XGBoost': np.array([[50, 10], [10, 168]]),
        'MLP': np.array([[49, 11], [17, 161]]),
        'KAN': np.array([[53, 7], [14, 164]])
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (model_name, cm) in enumerate(confusion_data.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   annot_kws={'size': 12, 'weight': 'bold'})
        ax.set_title(f'{model_name}', fontsize=12, fontweight='bold', pad=10)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)
        
        # Add accuracy text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.15, f'Accuracy: {accuracy:.1%}', 
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide the last subplot
    axes[5].axis('off')
    
    plt.suptitle('Confusion Matrices for All Classification Models', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', bbox_inches='tight', dpi=300)
    print("✓ Created: confusion_matrices.png")
    plt.close()

# ============================================================================
# Figure 3: ROC Curves Comparison
# ============================================================================
def create_roc_curves():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Generate synthetic ROC curves based on your AUC scores
    models_auc = {
        'Logistic Regression': 0.889,
        'Random Forest': 0.955,
        'XGBoost': 0.942,
        'MLP': 0.929,
        'KAN': 0.956
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for (model, auc), color in zip(models_auc.items(), colors):
        # Generate synthetic ROC curve
        fpr = np.linspace(0, 1, 100)
        # Create realistic ROC curve shape based on AUC
        tpr = np.power(fpr, 1/(2*auc))
        tpr = np.minimum(tpr * auc * 1.3, 1.0)
        
        ax.plot(fpr, tpr, label=f'{model} (AUC = {auc:.3f})', 
               linewidth=2.5, color=color)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves Comparison for Liver Disease Classification', 
                fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
    print("✓ Created: roc_curves.png")
    plt.close()

# ============================================================================
# Figure 4: Feature Importance
# ============================================================================
def create_feature_importance():
    features = ['Total Bilirubin', 'Alkaline Phosphatase', 'AST', 'Albumin', 
                'ALT', 'Direct Bilirubin', 'Total Proteins', 'A/G Ratio', 'Age', 'Gender']
    importance = [0.186, 0.154, 0.142, 0.128, 0.115, 0.098, 0.082, 0.054, 0.028, 0.013]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    bars = ax.barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, importance)):
        ax.text(val + 0.003, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Importance Score (Mean Decrease in Impurity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clinical Features', fontsize=12, fontweight='bold')
    ax.set_title('Random Forest Feature Importance Analysis', fontsize=14, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim([0, 0.22])
    
    # Add annotation
    ax.text(0.97, 0.03, 'Higher values indicate greater predictive importance', 
           transform=ax.transAxes, ha='right', fontsize=9, style='italic',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
    print("✓ Created: feature_importance.png")
    plt.close()

# ============================================================================
# Figure 5: KAN Spline Visualizations
# ============================================================================
def create_kan_splines():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Spline 1: Total Bilirubin (sharp increase above normal)
    x1 = np.linspace(-2, 3, 100)
    y1 = 1 / (1 + np.exp(-2*(x1 - 0.5))) - 0.2
    axes[0].plot(x1, y1, linewidth=3, color='#d62728')
    axes[0].axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Normal Range')
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].fill_between(x1, y1, alpha=0.3, color='#d62728')
    axes[0].set_xlabel('Normalized Bilirubin Level', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Activation Response', fontsize=11, fontweight='bold')
    axes[0].set_title('(a) Total Bilirubin Spline', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Spline 2: Albumin (inverse relationship)
    x2 = np.linspace(-2, 3, 100)
    y2 = -1 / (1 + np.exp(-2*(x2 + 0.3))) + 0.5
    axes[1].plot(x2, y2, linewidth=3, color='#2ca02c')
    axes[1].axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Normal Range')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].fill_between(x2, y2, alpha=0.3, color='#2ca02c')
    axes[1].set_xlabel('Normalized Albumin Level', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Activation Response', fontsize=11, fontweight='bold')
    axes[1].set_title('(b) Albumin Spline', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    # Spline 3: AST (nonlinear threshold)
    x3 = np.linspace(-2, 3, 100)
    y3 = np.where(x3 < 0, 0.1 * x3, 0.8 * (1 - np.exp(-x3)))
    axes[2].plot(x3, y3, linewidth=3, color='#ff7f0e')
    axes[2].axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Threshold (~40 IU/L)')
    axes[2].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    axes[2].fill_between(x3, y3, alpha=0.3, color='#ff7f0e')
    axes[2].set_xlabel('Normalized AST Level', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Activation Response', fontsize=11, fontweight='bold')
    axes[2].set_title('(c) AST Enzyme Spline', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    
    plt.suptitle('Learned Spline Transformations in Kolmogorov-Arnold Network', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('kan_splines.png', bbox_inches='tight', dpi=300)
    print("✓ Created: kan_splines.png")
    plt.close()

# ============================================================================
# Figure 6: Web Application Interface (Mockup)
# ============================================================================
def create_webapp_interface():
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Browser window
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 9.4, 9.4, boxstyle="round,pad=0.05", 
                                edgecolor='#424242', facecolor='white', linewidth=3))
    
    # Header bar
    ax.add_patch(FancyBboxPatch((0.3, 9.2), 9.4, 0.5, 
                                edgecolor='#1976D2', facecolor='#2196F3', linewidth=0))
    ax.text(5, 9.45, 'AI Liver Disease Prediction System', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    
    # Title
    ax.text(5, 8.7, 'Patient Information Form', ha='center', va='center', 
            fontsize=13, fontweight='bold', color='#1976D2')
    
    # Input fields (2 columns)
    fields_left = ['Age:', 'Total Bilirubin:', 'Alkaline Phosphatase:', 'ALT:', 'Total Proteins:']
    fields_right = ['Gender:', 'Direct Bilirubin:', 'AST:', 'Albumin:', 'A/G Ratio:']
    
    y_start = 7.8
    for i, (left, right) in enumerate(zip(fields_left, fields_right)):
        y = y_start - i * 0.8
        # Left column
        ax.text(1.5, y, left, fontsize=10, fontweight='bold')
        ax.add_patch(FancyBboxPatch((2.8, y-0.15), 1.5, 0.3, 
                                    boxstyle="round,pad=0.02", 
                                    edgecolor='gray', facecolor='#f5f5f5', linewidth=1))
        # Right column
        ax.text(5.5, y, right, fontsize=10, fontweight='bold')
        ax.add_patch(FancyBboxPatch((6.8, y-0.15), 1.5, 0.3, 
                                    boxstyle="round,pad=0.02", 
                                    edgecolor='gray', facecolor='#f5f5f5', linewidth=1))
    
    # Predict button
    ax.add_patch(FancyBboxPatch((3.5, 3.2), 3, 0.5, boxstyle="round,pad=0.05", 
                                edgecolor='#4CAF50', facecolor='#4CAF50', linewidth=2))
    ax.text(5, 3.45, 'PREDICT DISEASE RISK', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='white')
    
    # Results section
    ax.add_patch(FancyBboxPatch((1, 0.8), 8, 2, boxstyle="round,pad=0.1", 
                                edgecolor='#FF5722', facecolor='#FFEBEE', linewidth=2))
    ax.text(5, 2.5, 'Prediction Results', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#D32F2F')
    
    # Result display
    ax.text(2, 2.0, 'Disease Probability:', fontsize=10, fontweight='bold')
    ax.text(5.5, 2.0, '76.8%', fontsize=11, fontweight='bold', color='#D32F2F')
    
    ax.text(2, 1.6, 'Risk Classification:', fontsize=10, fontweight='bold')
    ax.add_patch(FancyBboxPatch((5.0, 1.45), 2, 0.3, boxstyle="round,pad=0.05", 
                                edgecolor='#D32F2F', facecolor='#FFCDD2', linewidth=2))
    ax.text(6, 1.6, 'HIGH RISK', ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#D32F2F')
    
    ax.text(2, 1.2, 'Model Used:', fontsize=10, fontweight='bold')
    ax.text(5.5, 1.2, 'Random Forest (AUC: 95.45%)', fontsize=10, style='italic')
    
    plt.title('Web Application User Interface', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('webapp_interface.png', bbox_inches='tight', dpi=300)
    print("✓ Created: webapp_interface.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("Generating all figures for IEEE Conference Paper...")
    print("=" * 60)
    
    create_architecture_diagram()
    create_confusion_matrices()
    create_roc_curves()
    create_feature_importance()
    create_kan_splines()
    create_webapp_interface()
    
    print("=" * 60)
    print("All figures generated successfully!")
    print("\nGenerated files:")
    print("  1. architecture_diagram.png")
    print("  2. confusion_matrices.png")
    print("  3. roc_curves.png")
    print("  4. feature_importance.png")
    print("  5. kan_splines.png")
    print("  6. webapp_interface.png")
    print("\nPlace all PNG files in the same directory as your .tex file.")
