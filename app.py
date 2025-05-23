import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os

# Set page config to wide layout
st.set_page_config(layout="wide")

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

def e(x,
      a1=700, m1=800, sigma1=350,   # Paramètres du pic initial
      a2=900, m2=2000, sigma2=500,  # Paramètres du creux central
      L=9000, k=0.002, m3=9000      # Paramètres de la sigmoïde finale
    ):
    # x est le niveau réel d'apprentissage
    # Ajustez les paramètres pour adapter la courbe à vos besoins
    bump = a1 * np.exp(-((x - m1) ** 2) / (2 * sigma1 ** 2))
    dip  = a2 * np.exp(-((x - m2) ** 2) / (2 * sigma2 ** 2))
    sigmoid = L / (1 + np.exp(-k * (x - m3)))
    return x + bump - dip + sigmoid

def e(x, a1, m1, sigma1, a2, m2, sigma2, L, k, m3):
    bump = a1 * np.exp(-((x - m1) ** 2) / (2 * sigma1 ** 2))
    dip  = a2 * np.exp(-((x - m2) ** 2) / (2 * sigma2 ** 2))
    sigmoid = L / (1 + np.exp(-k * (x - m3))) - L
    return x + bump - dip + sigmoid

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
    st.sidebar.header("Paramètres")

    # Section "Niveau de référence"
    st.sidebar.subheader("Niveau de référence")
    R0 = st.sidebar.slider("R0 (Niveau de référence initial)", 100, 2000, 1000)
    k = st.sidebar.slider("k (Taux de croissance exponentielle)", 0.01, 0.1, 0.05)

    # Section "Courbe d'apprentissage"
    st.sidebar.subheader("Courbe d'apprentissage")
    f0 = st.sidebar.slider("f0 (Niveau initial de compétence)", 10, 500, 100)
    beta = st.sidebar.slider("beta (Taux d'apprentissage)", 0.01, 0.99, 0.5)

    # Section "Courbe d'auto-évaluation"
    st.sidebar.subheader("Courbe d'auto-évaluation oscillante")
    alpha = st.sidebar.slider("alpha (Facteur de proportionnalité)", 0.1, 2.0, 1.0)
    omega = st.sidebar.slider("omega (Fréquence des oscillations)", 0.1, 5.0, 1.0)

    st.sidebar.subheader("Courbe d'auto-évaluation")

    a1 = st.sidebar.slider("a1 (amplitude du pic initial)", 0, 2000, 700)
    m1 = st.sidebar.slider("m1 (centre du pic initial)", 0, 5000, 800)
    sigma1 = st.sidebar.slider("sigma1 (largeur du pic initial)", 1, 2000, 350)
    
    a2 = st.sidebar.slider("a2 (amplitude du creux)", 0, 2000, 900)
    m2 = st.sidebar.slider("m2 (centre du creux)", 0, 5000, 2000)
    sigma2 = st.sidebar.slider("sigma2 (largeur du creux)", 1, 2000, 500)
    
    L = st.sidebar.slider("L (amplitude de la sigmoïde finale)", 0, 5000, 800)
    k = st.sidebar.slider("k (pente de la sigmoïde)", 0.0001, 0.01, 0.002, step=0.0001, format="%.4f")
    m3 = st.sidebar.slider("m3 (centre de la sigmoïde)", 0, 15000, 9000)

    # Section "Représentation"
    st.sidebar.subheader("Représentation")
    x_range = st.sidebar.slider("Intervalle de temps (x)", 1, 100, (1, 100))
    y_range = st.sidebar.slider("Intervalle de niveau d'apprentissage (y)", 0, 10000, (0, 10000))

    # Générer les valeurs x et y
    x = np.linspace(x_range[0], x_range[1], 500)
    y = np.linspace(y_range[0], y_range[1], 500)

    # Plot de la fonction g
    st.subheader("Variation de l'auto-évaluation en fonction de l'apprentissage réel")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    g_values = g(y, alpha, omega)
    e_values = e(y, a1, m1, sigma1, a2, m2, sigma2, L, k, m3)
    ax1.plot(y, g_values, label=r'$g(x) = \text{niveau auto-évalué en fonction du niveau réel}$', color='purple')
    ax1.plot(y, y, label=r'$g(x) = x = \text{auto-évaluation réaliste}$', color='gray', linestyle='--')
    ax1.plot(y, e_values, label=r'$e(x) = \text{auto-évaluation en fonction de la compétence réelle}$', color='orange', linewidth=2)

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
        
        st.markdown("**g(y) = auto-évaluation en fonction de l'apprentissage**")
        st.latex(r'''
        g(y) = y + \alpha \cdot \sin(\omega \cdot y)
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

