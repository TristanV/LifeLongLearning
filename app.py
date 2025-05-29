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
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Masques pour les segments
    mask_spline1 = (x >= 0) & (x <= R/4)
    mask_spline2 = (x > R/4) & (x <= R/2)
    mask_sigmoid = (x > R/2) & (x <= R)
    mask_linear = x > R
    
    # 1. Première partie de la spline (0 ≤ x ≤ R/4)
    if np.any(mask_spline1):
        t = np.pi * x[mask_spline1] / R
        y[mask_spline1] = (24*R/np.pi**3) * t**2 * (3 - 4*t)
    
    # 2. Seconde partie de la spline (R/4 < x ≤ R/2)
    if np.any(mask_spline2):
        t = np.pi * (x[mask_spline2] - R/4) / R
        y[mask_spline2] = R - (24*R/np.pi**3) * t**2 * (3 + 4*t)
    
    # 3. Sigmoïde ajustée (R/2 < x ≤ R)
    if np.any(mask_sigmoid):
        k = pente_sigmoide * (1 + np.sqrt(5))  # Facteur de calibration
        z = (2 * x[mask_sigmoid]/R) - 1
        y[mask_sigmoid] = R/4 + (3*R/4)/(1 + np.exp(-k*z))
    
    # 4. Régime linéaire (x > R)
    y[mask_linear] = x[mask_linear]
    
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

    st.sidebar.image("static/images/logo_lifelonglearning_v3.png", use_container_width =True)
    
    st.sidebar.header("Paramètres")

    with st.sidebar.expander("Niveau de référence", expanded=False):
        R0 = st.slider("R0 (Niveau de référence initial)", 100, 5000, 4000, step=100)
        k = st.slider("k (Taux de croissance exponentielle du niveau de référence)", 0.001, 0.1, 0.005)

    with st.sidebar.expander("Courbe d'auto-évaluation", expanded=False):
        pente_sigmoide = st.slider("pente_sigmoide (raideur sigmoïde)", 0.01, 5.0, 1.0, step=0.01)
    


    # Section "Courbe d'apprentissage"
    with st.sidebar.expander("Courbe d'apprentissage", expanded=False):
        f0 = st.slider("f0 (Niveau initial de compétence)", 10, 500, 100)
        beta = st.slider("beta (Taux d'apprentissage)", 0.01, 0.99, 0.3)

    # Section "Courbe d'auto-évaluation"
    with st.sidebar.expander("Courbe d'auto-évaluation oscillante", expanded=False):
        alpha = st.slider("alpha (Facteur de proportionnalité)", 0.1, 2.0, 1.0)
        omega = st.slider("omega (Fréquence des oscillations)", 0.1, 5.0, 1.0)


    # Section "Représentation"
    with st.sidebar.expander("Axes", expanded=False):
        x_range = st.slider("Intervalle de temps", 1, 100, (1, 100))
        y_range = st.slider("Intervalle de niveau d'apprentissage", 0, 10000, (0, 5000))

    # Générer les valeurs x et y
    x = np.linspace(x_range[0], x_range[1], 500) #temps
    y = np.linspace(y_range[0], y_range[1], 500) #niveau d'apprentissage

    # Plot de la fonction g
    st.subheader("Variation de l'auto-évaluation en fonction de l'apprentissage réel")
    fig1, ax1 = plt.subplots(figsize=(12, 6)) 
    evalearn_values = evalearn(y, R0, pente_sigmoide) 
    ax1.plot(y, y, label=r'$\text{niveau auto-évalué = niveau réel}$', color='gray', linestyle='--')
    ax1.plot(y, evalearn_values, label=r'$\mathrm{evalearn}(x)$', color='blue')
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
        
        st.latex(r"""
        \text{evalearn}(x) = 
        \begin{cases} 
        \frac{24R}{\pi^3} \left(\frac{\pi x}{R}\right)^2 \left(3 - 4\frac{\pi x}{R}\right) & \text{si } 0 \leq x \leq \frac{R}{4}, \\
        R - \frac{24R}{\pi^3} \left(\frac{\pi(x - R/4)}{R}\right)^2 \left(3 + 4\frac{\pi(x - R/4)}{R}\right) & \text{si } \frac{R}{4} < x \leq \frac{R}{2}, \\
        \frac{R}{4} + \frac{3R}{4} \cdot \frac{1}{1 + e^{-k\left(\frac{2x}{R} - 1\right)}} & \text{si } \frac{R}{2} < x \leq R, \\
        x & \text{si } x > R.
        \end{cases}
        """)

        
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

