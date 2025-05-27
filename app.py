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

def e(x, a1, m1, sigma1, a2, m2, sigma2, L, k, m3):
    bump = a1 * np.exp(-((x - m1) ** 2) / (2 * sigma1 ** 2))
    dip  = a2 * np.exp(-((x - m2) ** 2) / (2 * sigma2 ** 2))
    sigmoid = L / (1 + np.exp(-k * (x - m3))) - L
    return x + bump - dip + sigmoid 

def bezier_cubic(t, P0, P1, P2, P3):
    return (
        (1-t)**3 * P0 +
        3*(1-t)**2 * t * P1 +
        3*(1-t) * t**2 * P2 +
        t**3 * P3
    )

def e_composite(x, x_creux, y_c1, y_c2, slope=0.01):
    """
    x: array of real skill values
    x_creux: abscisse du point de confiance minimale (creux)
    y_c1, y_c2: contrôlent la forme de la Bézier jusqu'au creux
    slope: raideur de la transition sigmoïde
    """
    x = np.array(x)
    e_vals = np.zeros_like(x)
    # Phase 1: Bézier sur [0, x_creux]
    t = np.clip(x / x_creux, 0, 1)
    B = x_creux * bezier_cubic(
        t,
        0,
        y_c1,  # Surévaluation initiale (en proportion de x_creux)
        y_c2,  # Profondeur du creux (en proportion de x_creux)
        1      # Retour à la diagonale au point d'inflexion (x_creux, x_creux)
    )
    # Valeur au creux pour raccord
    e_creux = B[-1] if isinstance(B, np.ndarray) else B
    # Phase 2: Sigmoïde vers y=x pour x > x_creux
    mask = x > x_creux
    if np.any(mask):
        x_sig = x[mask]
        # Sigmoïde croissante de 0 (au creux) à 1 (pour x>>x_creux)
        s = 1 / (1 + np.exp(-slope * (x_sig - x_creux)))
        # La sigmoïde relie e_creux à x
        e_vals[mask] = (1-s) * e_creux + s * x_sig
    # Phase 1: Bézier
    e_vals[~mask] = B[~mask]
    return e_vals

def evalearn(x, R, maximum_local, minimum_local, pente_sigmoide):
    """
    Modélise la courbe d'auto-évaluation avec trois zones distinctes.
    
    Paramètres :
        x (float ou np.ndarray) : Valeur(s) de compétence réelle
        R (float) : Niveau de référence à atteindre
        maximum_local (float) : Valeur du maximum local en x=R/4
        minimum_local (float) : Valeur du minimum local en x=R/2
        pente_sigmoide (float) : Contrôle la raideur de la transition sigmoïde
    
    Retourne :
        float ou np.ndarray : Valeur(s) de l'auto-évaluation e(x)
    """
    
    x = np.array(x, dtype=float)
    y = np.zeros_like(x)
    
    # Zone 1 : Comportement sinusoïdal-linéaire [0, R/2]
    mask1 = (x >= 0) & (x <= R/2)
    if np.any(mask1):
        # Résolution du système pour a et b :
        # e(R/4) = a*1 + b*(R/4) = maximum_local
        # e(R/2) = a*0 + b*(R/2) = minimum_local
        b = (2 * minimum_local) / R
        a = maximum_local - b * R/4
        
        # Application de la formule composite
        y[mask1] = a * np.sin(2 * np.pi * x[mask1]/R) + b * x[mask1]
    
    # Zone 2 : Demi-sigmoïde ajustée (R/2, R]
    mask2 = (x > R/2) & (x <= R)
    if np.any(mask2):
        # Centrage de la sigmoïde sur 3R/4
        x_centre = 0.75 * R
        
        # Fonction sigmoïde standard
        sigmoid = lambda z: 1 / (1 + np.exp(-z))
        
        # Ajustement de l'échelle et du décalage
        y[mask2] = minimum_local + (R - minimum_local) * sigmoid(
            pente_sigmoide * (x[mask2] - x_centre)/(R/2)
        )
    
    # Zone 3 : Alignement parfait (x > R)
    mask3 = (x > R)
    if np.any(mask3):
        y[mask3] = x[mask3]
    
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

        a1 = st.slider("a1 (amplitude du pic initial)", 0, 2000, 700)
        m1 = st.slider("m1 (centre du pic initial)", 0, 5000, 800)
        sigma1 = st.slider("sigma1 (largeur du pic initial)", 1, 2000, 350)
        
        a2 = st.slider("a2 (amplitude du creux)", 0, 2000, 900)
        m2 = st.slider("m2 (centre du creux)", 0, 5000, 2000)
        sigma2 = st.slider("sigma2 (largeur du creux)", 1, 2000, 500)
        
        L = st.slider("L (amplitude de la sigmoïde finale)", 0, 5000, 800)
        k = st.slider("k (pente de la sigmoïde)", 0.0001, 0.01, 0.002, step=0.0001, format="%.4f")
        m3 = st.slider("m3 (centre de la sigmoïde)", 0, 15000, 9000)
                             
    with st.sidebar.expander("Auto-évaluation réaliste (Dunning-Kruger + expertise)", expanded=True):
        x_creux = st.slider("x_creux (niveau réel au creux)", 100, 9000, 2000)
        y_c1 = st.slider("y_c1 (contrôle sur-début, >1 pour surévaluation)", 0.5, 2.0, 1.2, step=0.01)
        y_c2 = st.slider("y_c2 (profondeur creux, <1 pour sous-évaluation)", -1.0, 1.0, 0.2, step=0.01)
        slope = st.slider("slope (raideur de la remontée finale)", 0.001, 0.05, 0.01, step=0.001)

    with st.sidebar.expander("Courbe d'auto-évaluation (modèle evalearn)", expanded=False):
        maximum_local = st.slider("maximum_local (valeur max local en x=R/4)", 0, 10000, 3000)
        minimum_local = st.slider("minimum_local (valeur min local en x=R/2)", 0, 10000, 1000)
        pente_sigmoide = st.slider("pente_sigmoide (raideur sigmoïde)", 0.01, 5.0, 1.0, step=0.01)

    
    # Section "Niveau de référence"
    with st.sidebar.expander("Niveau de référence", expanded=False):
        R0 = st.slider("R0 (Niveau de référence initial)", 100, 2000, 1000)
        k = st.slider("k (Taux de croissance exponentielle)", 0.01, 0.1, 0.05)

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
    e_values = e(y, a1, m1, sigma1, a2, m2, sigma2, L, k, m3)
    e_vals = e_composite(y, x_creux, y_c1, y_c2, slope)
    evalearn_values = evalearn(y, R0, maximum_local, minimum_local, pente_sigmoide)
    ax1.plot(y, g_values, label=r'$g(x) = \text{niveau auto-évalué en fonction du niveau réel}$', color='purple')
    ax1.plot(y, y, label=r'$g(x) = x = \text{auto-évaluation réaliste}$', color='gray', linestyle='--')
    ax1.plot(y, e_values, label=r'$e(x) = \text{auto-évaluation en fonction de la compétence réelle}$', color='pink', linewidth=2)
    ax1.plot(y, e_vals, label=r'$e_{\text{réaliste}}(x)$', color='orange', linewidth=2)
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
        e(x) = x + a_1 \exp\left(-\frac{(x - m_1)^2}{2 \sigma_1^2}\right)
               - a_2 \exp\left(-\frac{(x - m_2)^2}{2 \sigma_2^2}\right)
               + \left[ \frac{L}{1 + e^{-k(x - m_3)}} - L \right]
        ''')

        st.latex(r'''
        e(x) = 
        \begin{cases}
        x_{\text{creux}} \cdot \mathrm{Bézier}_3(t), & x \leq x_{\text{creux}} \\
        (1-s)\,e_{\text{creux}} + s\,x, & x > x_{\text{creux}},\quad s = \frac{1}{1 + e^{-\alpha(x - x_{\text{creux}})}}
        \end{cases}
        \qquad t = \frac{x}{x_{\text{creux}}}
        ''')
        
        st.markdown("**g(y) = auto-évaluation en fonction de l'apprentissage**")
        st.latex(r'''
        g(y) = y + \alpha \cdot \sin(\omega \cdot y)
        ''')

        st.markdown("**evalearn(x) = auto-évaluation (par parties)**")
        st.latex(r'''
        \mathrm{evalearn}(x) =
        \begin{cases}
        a\, \sin\left(2\pi \frac{x}{R}\right) + b\, x, & 0 \leq x \leq \frac{R}{2} \\
        \text{sigmoïde croissante}, & \frac{R}{2} < x \leq R \\
        x, & x > R
        \end{cases}
        ''')
        st.markdown(r'''
        où :
        - $a = \mathrm{maximum\_local} - \frac{R}{4} \cdot b$, $b = \frac{2\,\mathrm{minimum\_local}}{R}$
        - sigmoïde centrée sur $x_c = \frac{3R}{4}$ et raideur contrôlée par ``pente_sigmoide`` :
        ''')
        st.latex(r'''
        \mathrm{sigmoide}(x) = \mathrm{minimum\_local} + (R - \mathrm{minimum\_local}) \cdot \left[ \frac{1}{1 + e^{-\mathrm{pente\_sigmoide} \cdot \frac{x-x_c}{R/2}}} \right]
        ''')
        
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

