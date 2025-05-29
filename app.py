import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page config to wide layout
st.set_page_config(
    page_title="LifeLongLearning",
    page_icon=":superhero:",
    layout="wide",
    initial_sidebar_state="expanded" # collapsed | expanded
)
 
# Lire le contenu du README.md
def get_readme_content():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return "README.md non trouvé."

# Définir les fonctions (inchangées)
def R(x, R0, k):
    return R0 * np.exp(k * x)

def evalearn(x, R, pente_sigmoide):
    """
    Modélisation précise de la courbe d'auto-évaluation avec contraintes strictes.
    
    Paramètres :
        x (float/array) : Compétence réelle
        R (float) : Niveau de référence
        pente_sigmoide (float) : Contrôle la raideur de la transition
    
    Retourne :
        float/array : Valeur(s) de l'auto-évaluation
    """
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Masques pour les zones
    mask_sin = (x >= 0) & (x <= R/2)
    mask_sig = (x > R/2) & (x <= R)
    mask_lin = (x > R)
    
    # 1. Segment sinusoïdal (0 ≤ x ≤ R/2)
    if np.any(mask_sin):
        k = 4 * np.pi / R
        y[mask_sin] = (R/4) * (1 - np.cos(k * x[mask_sin]))
    
    # 2. Demi-sigmoïde ajustée (R/2 < x ≤ R)
    if np.any(mask_sig):
        # Facteur de correction pour le point d'inflexion
        phi = (1 + np.sqrt(5))/2  # Nombre d'or
        k = pente_sigmoide * phi
        
        # Calcul de la sigmoïde normalisée
        z = (x[mask_sig] - R/2)/(R/2)
        sigmoid = 1 / (1 + np.exp(-k*(z - 0.5)))
        
        y[mask_sig] = R/4 + (3*R/4)*sigmoid
    
    # 3. Alignement linéaire (x > R)
    y[mask_lin] = x[mask_lin]
    
    return y


def f(x, R0, k, f0, beta):
    return R(x, R0, k) - (R0 - f0) * x**(-beta)

def A(x, R0, k, f0, beta, alpha):
    return alpha * (R(x, R0, k) - f(x, R0, k, f0, beta))

def g(y, alpha, omega):
    return y + alpha * np.sin(omega * y)

def h(x, R0, k, f0, beta, alpha, omega):
    return f(x, R0, k, f0, beta) + A(x, R0, k, f0, beta, alpha) * np.sin(omega * x)

# Créer les onglets
tabs = st.tabs(["Apprentissage", "Information"])

with tabs[0]:
    st.title("LifeLongLearning")

    # Sidebar for parameters

    st.sidebar.image("static/images/logo_lifelonglearning_v1.png", use_container_width =True)
    
    st.sidebar.header("Paramètres")


    with st.sidebar.expander("Courbe d'auto-évaluation", expanded=False):

        k = st.slider("k (pente de la sigmoïde)", 0.0001, 0.01, 0.002, step=0.0001, format="%.4f")
        pente_sigmoide = st.slider("pente_sigmoide (raideur sigmoïde)", 0.01, 5.0, 1.0, step=0.01)

    
    # Section "Niveau de référence"
    with st.sidebar.expander("Niveau de référence", expanded=False):
        R0 = st.slider("R0 (Niveau de référence initial)", 100, 2000, 1000)
        k = st.slider("k (Taux de croissance exponentielle du niveau de référence)", 0.01, 0.1, 0.05)

    # Section "Courbe d'apprentissage"
    with st.sidebar.expander("Courbe d'apprentissage", expanded=False):
        f0 = st.slider("f0 (Niveau initial de compétence)", 10, 500, 100)
        beta = st.slider("beta (Taux d'apprentissage)", 0.01, 0.99, 0.5)

    # Section "Courbe d'auto-évaluation"
    with st.sidebar.expander("Courbe d'auto-évaluation oscillante", expanded=False):
        alpha = st.slider("alpha (Facteur de proportionnalité)", 0.1, 2.0, 1.0)
        omega = st.slider("omega (Fréquence des oscillations)", 0.1, 5.0, 1.0)


    # Section "Représentation"
    with st.sidebar.expander("Axes", expanded=False):
        x_range = st.slider("Intervalle de temps (x)", 1, 100, (1, 100))
        y_range = st.slider("Intervalle de niveau d'apprentissage (y)", 0, 10000, (0, 10000))

    # Générer les valeurs x et y
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)

    # Plot de la fonction g
    st.subheader("Variation de l'auto-évaluation en fonction de l'apprentissage réel")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    g_values = g(y, alpha, omega)
    evalearn_values = evalearn(y, R0, pente_sigmoide)
    ax1.plot(y, g_values, label=r'$g(x) = \text{niveau auto-évalué en fonction du niveau réel}$', color='purple')
    ax1.plot(y, y, label=r'$g(x) = x = \text{auto-évaluation réaliste}$', color='gray', linestyle='--')
    ax1.plot(y, evalearn_values, label=r'$\mathrm{evalearn}(x)$', color='blue', linewidth=2)
    ax1.set_title('Niveau auto-évalué en fonction du niveau réel')
    ax1.set_xlabel('Niveau d\'apprentissage réel')
    ax1.set_ylabel('Niveau d\'apprentissage auto-évalué')
    ax1.set_xlim(y_range[0], y_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig1, use_container_width=True)

    # Plot des fonctions pour R(x) constant et exponentiel
    st.subheader("Variation de l'auto-évaluation et de l'apprentissage réel en fonction du temps")
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))

    # Objectif constant
    R_constant_values = R0 * np.ones_like(x)
    f_constant_values = f(x, R0, 0, f0, beta)
    axs[0].plot(x, R_constant_values, label=r'$R(x)$', color='red', linestyle='--')
    axs[0].plot(x, f_constant_values, label=r'$f(x)$', color='green')
    axs[0].plot(x, h(x, R0, 0, f0, beta, alpha, omega), label=r'$h(x)$', color='blue')
    axs[0].set_title('Niveau d\'apprentissage avec objectif constant')
    axs[0].set_xlabel('Temps $x$')
    axs[0].set_ylabel('Niveau d\'apprentissage')
    axs[0].set_xlim(x_range[0], x_range[1])
    axs[0].set_ylim(y_range[0], y_range[1])
    axs[0].legend()
    axs[0].grid(True)

    # Objectif exponentiel
    R_exponential_values = R(x, R0, k)
    f_exponential_values = f(x, R0, k, f0, beta)
    axs[1].plot(x, R_exponential_values, label=r'$R(x)$', color='red', linestyle='--')
    axs[1].plot(x, f_exponential_values, label=r'$f(x)$', color='green')
    axs[1].plot(x, h(x, R0, k, f0, beta, alpha, omega), label=r'$h(x)$', color='blue')
    axs[1].set_title('Niveau d\'apprentissage avec objectif croissant exponentiellement')
    axs[1].set_xlabel('Temps $x$')
    axs[1].set_ylabel('Niveau d\'apprentissage')
    axs[1].set_xlim(x_range[0], x_range[1])
    axs[1].set_ylim(y_range[0], y_range[1])
    axs[1].legend()
    axs[1].grid(True)

    st.pyplot(fig, use_container_width=True)

    # Affichage des équations
    st.subheader("Équations des fonctions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**y = f(x) = apprentissage au fil du temps**")
        st.latex(r'''
        f(x) = R(x) - (R_0 - f_0) \cdot x^{-\beta}
        ''')

        st.markdown("**e(x) = auto-évaluation en fonction de la compétence réelle**")
        st.latex(r'''
        g(y) = y + \alpha \cdot \sin(\omega \cdot y)
        ''')

        st.markdown("**evalearn(x) = auto-évaluation (par parties)**")
        #à-compléter
        
    with col2:
        st.markdown("**A(x) = amplitude des oscillations**")
        st.latex(r'''
        A(x) = \alpha \cdot (R(x) - f(x))
        ''')
        st.markdown("**h(x) = g(f(x)) = auto-évaluation au fil du temps**")
        st.latex(r'''
        h(x) = f(x) + A(x) \cdot \sin(\omega \cdot x)
        ''')

with tabs[1]:
    st.title("Information")
    st.markdown(get_readme_content())

