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
def Ref(x, R, k):
    return R * np.exp(k * x)


def evalearn(x, R):
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)
    
    # Masques pour les segments
    mask_spline1 = (x >= 0) & (x <= R/4)
    mask_spline2 = (x > R/4) & (x <= R/2)
    mask_poly = (x > R/2) & (x <= R)
    mask_linear = x > R
    
    # 1. Spline cubique (0 ≤ x ≤ R/2)
    if np.any(mask_spline1):
        x1 = x[mask_spline1]
        y[mask_spline1] = (-64/(R**2))*x1**3 + (24/R)*x1**2
    
    if np.any(mask_spline2):
        x2 = x[mask_spline2]
        y[mask_spline2] = (32/(R**2))*x2**3 - (36/R)*x2**2 + 12*x2 - (3*R)/4
    
    # 2. Segment polynomial (R/2 < x ≤ R)
    if np.any(mask_poly):
        t = x[mask_poly] - R/2
        a = 20 / R**3
        b = -28 / R**2
        c = 12 / R
        y[mask_poly] = a*t**4 + b*t**3 + c*t**2 + R/4
    
    # 3. Régime linéaire (x > R)
    y[mask_linear] = x[mask_linear]
    
    return y



def f(x, R, k, f0, beta):
    return Ref(x, R, k) - (R - f0) * x**(-beta)

def h(x, R, k, f0, beta):
    return evalearn(f(x, R, k, f0, beta), R)
    
# Créer les onglets
tabs = st.tabs(["Apprentissage", "Information"])

with tabs[0]:
    st.title("LifeLongLearning")

    # Sidebar for parameters

    st.sidebar.image("static/images/logo_lifelonglearning_v3.png", use_container_width =True)
    
    st.sidebar.header("Paramètres")

    with st.sidebar.expander("Niveau de référence", expanded=False):
        R0 = st.slider("R0 (Niveau de référence initial)", 100, 5000, 4000, step=100)
        k = st.slider("k (Taux de croissance exponentielle du niveau de référence)", 0.01, 0.1, 0.01)

    with st.sidebar.expander("Courbe d'auto-évaluation", expanded=False):
        pente_sigmoide = st.slider("non utilisé", 0.01, 5.0, 1.0, step=0.01)
    

    # Section "Courbe d'apprentissage"
    with st.sidebar.expander("Courbe d'apprentissage", expanded=False):
        f0 = st.slider("f0 (Niveau initial de compétence)", 10, 500, 100)
        beta = st.slider("beta (Taux d'apprentissage)", 0.01, 0.99, 0.05)

    with st.sidebar.expander("Axes", expanded=False):
        x_range = st.slider("Intervalle de temps", 1, 100, (1, 100), step=1)
        y_range = st.slider("Intervalle de niveau d'apprentissage", 0, 10000, (0, 5000), step=100)

    # Générer les valeurs x et y
    x = np.linspace(x_range[0], x_range[1], 500) #temps
    y = np.linspace(y_range[0], y_range[1], 500) #niveau d'apprentissage

    # Plot de la fonction evalearn
    st.subheader("Variation de l'auto-évaluation en fonction de l'apprentissage réel")
    fig1, ax1 = plt.subplots(figsize=(12, 6)) 
    evalearn_values = evalearn(y, R0) 
    ax1.plot(y, y, label=r'$\text{niveau auto-évalué = niveau réel}$', color='gray', linestyle='--')
    ax1.axhline(y=R0, color='red', linestyle=':', linewidth=1, label=r'$y = R_0$')
    ax1.axvline(x=R0, color='red', linestyle=':', linewidth=1, label=r'$x = R_0$')
    ax1.plot(y, evalearn_values, label=r'$\mathrm{evalearn}(x)$', color='blue')
    ax1.set_title('Niveau auto-évalué en fonction du niveau réel')
    ax1.set_xlabel('Niveau d\'apprentissage réel')
    ax1.set_ylabel('Niveau d\'apprentissage auto-évalué')
    ax1.set_xlim(y_range[0], y_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.legend(loc='lower right')  # <-- légende en bas à droite
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
    axs[0].plot(x, h(x, R0, 0, f0, beta), label=r'$h(x)$', color='blue')
    axs[0].set_title('Niveau d\'apprentissage avec objectif constant')
    axs[0].set_xlabel('Temps $x$')
    axs[0].set_ylabel('Niveau d\'apprentissage')
    axs[0].set_xlim(x_range[0], x_range[1])
    axs[0].set_ylim(y_range[0], y_range[1])
    axs[0].legend()
    axs[0].grid(True)

    # Objectif exponentiel
    R_exponential_values = Ref(x, R0, k)
    f_exponential_values = f(x, R0, k, f0, beta)
    axs[1].plot(x, R_exponential_values, label=r'$R(x)$', color='red', linestyle='--')
    axs[1].plot(x, f_exponential_values, label=r'$f(x)$', color='green')
    axs[1].plot(x, h(x, R0, k, f0, beta), label=r'$h(x)$', color='blue')
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


        st.markdown("**evalearn(x, R) = auto-évaluation (par parties) en fonction du niveau réel x et d'un objectif de référence R**")
        
        st.latex(r"""
        \text{evalearn}(x,R) = 
        \begin{cases} 
        -\frac{64}{R^2}x^3 + \frac{24}{R}x^2 & x \in [0, \frac{R}{4}], \\
        \frac{32}{R^2}x^3 - \frac{36}{R}x^2 + 12x - \frac{3R}{4} & x \in (\frac{R}{4}, \frac{R}{2}], \\
        \frac{20}{R^3}t^4 - \frac{28}{R^2}t^3 + \frac{12}{R}t^2 + \frac{R}{4} & t = x - \frac{R}{2},\ x \in (\frac{R}{2}, R], \\
        x & x > R.
        \end{cases}
        """)

        
    with col2:
        st.markdown("**y = f(x, R_0, f_0, beta) = apprentissage au fil du temps**")
        st.latex(r'''
        f(x) = R(x) - (R_0 - f_0) \cdot x^{-\beta}
        ''')
        
        st.markdown("**h(x,R) = evalearn(f(x,R)) = auto-évaluation au fil du temps**")
        st.latex(r"""
        h(x, R, k, f_0, \beta) = \mathrm{evalearn}\left( R(x) - (R_0 - f_0)x^{-\beta},\ R \right)
        """)

with tabs[1]:
    st.title("Information")
    st.markdown(get_readme_content())

