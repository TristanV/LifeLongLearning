import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
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

# Fonction déterminant le niveau de référence à atteindre en fonction du temps t, d'un niveau de référence initial R0 et d'un exposant de croissance exponentielle k
def Ref(t, R0, k):
    return R0 * (np.exp(k * t) - k * t)
    
# Fonction décrivant le niveau de compétence auto-évalué en fonction du niveau de compétence réel et en fonction d'un niveau de référence R à atteindre.
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


# compétence acquise en fonction du temps et d'un taux d'apprentissage, avec un objectif de référence éventuellement variable
def f(t, R0, k, f0, beta):
    return Ref(t, R0, k) - (R0 - f0) * t**(-beta)

# wrapper pour calculer evalearn en fonction du temps t , c'est à dire en calculant au préalable la compétence f(t) et la référence Ref(t)
def h(t, R0, k, f0, beta):
    x = f(t, R0, k, f0, beta)
    R = Ref(t, R0, k)
    y = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        y[i] = evalearn(x[i], R[i])
    return y
   # return evalearn(f(t, R0, k, f0, beta), Ref(t, R0, k))
    
# Créer les onglets
tabs = st.tabs(["Apprentissage et compétences", "Apprentissage et auto-évaluation",  "Information"])

with tabs[0]:
    st.title("LifeLongLearning")
     # Sidebar for parameters

    st.sidebar.image("static/images/logo_lifelonglearning_v3.png", use_container_width =True)
    
    st.sidebar.header("Paramètres")

with tabs[1]:
    st.title("LifeLongLearning")

    # Sidebar for parameters

    st.sidebar.image("static/images/logo_lifelonglearning_v3.png", use_container_width =True)
    
    st.sidebar.header("Paramètres")

    with st.sidebar.expander("Niveau de référence", expanded=False):
        R0 = st.slider("R0 (Niveau de référence initial)", 100, 5000, 4000, step=100)
        k = st.slider("k (Taux de croissance exponentielle du niveau de référence)", 0.01, 0.1, 0.01)
    

    # Section "Courbe d'apprentissage"
    with st.sidebar.expander("Courbe d'apprentissage", expanded=False):
        f0 = st.slider("f0 (Niveau initial de compétence)", 10, 1000, 100)
        beta = st.slider("beta (Taux d'apprentissage)", 0.01, 0.99, 0.3)

    with st.sidebar.expander("Axes", expanded=False):
        x_range = st.slider("Intervalle de temps", 1, 1000, (1, 100), step=1)
        y_range = st.slider("Intervalle de niveau d'apprentissage", 0, 10000, (0, 5000), step=100)

    # ################################################################ 
    # Générer les valeurs x et y
    x = np.linspace(x_range[0], x_range[1], 500) #temps
    y = np.linspace(y_range[0], y_range[1], 500) #niveau d'apprentissage

    # Plot de la fonction evalearn
    st.subheader("Variation de l'auto-évaluation en fonction de l'apprentissage réel")
    fig1, ax1 = plt.subplots(figsize=(12, 6)) 
    evalearn_values = evalearn(y, R0) 
    ax1.plot(y, y, label=r'$\text{niveau auto-évalué e = niveau réel c}$', color='gray', linestyle='--')
    ax1.axhline(y=R0, color='red', linestyle=':', linewidth=1, label=r'$e = R_0$')
    ax1.axvline(x=R0, color='red', linestyle=':', linewidth=1, label=r'$c = R_0$')
    ax1.plot(y, evalearn_values, label=r'$\mathrm{evalearn}(c)$', color='orange')
    ax1.set_title('Niveau auto-évalué en fonction du niveau réel')
    ax1.set_xlabel('c = Niveau d\'apprentissage réel')
    ax1.set_ylabel('e = Niveau d\'apprentissage auto-évalué')
    ax1.set_xlim(y_range[0], y_range[1])
    ax1.set_ylim(y_range[0], y_range[1])
    ax1.legend(loc='lower right')  # <-- légende en bas à droite
    ax1.grid(True)
    st.pyplot(fig1, use_container_width=True)


    st.markdown("**e = evalearn(c, R) = auto-évaluation (par parties) en fonction du niveau réel c et d'un objectif de référence R**")
    
    st.latex(r"""
    \text{evalearn}(c,R) = 
    \begin{cases} 
    -\frac{64}{R^2}c^3 + \frac{24}{R}c^2 & c \in [0, \frac{R}{4}], \\
    \frac{32}{R^2}c^3 - \frac{36}{R}c^2 + 12c - \frac{3R}{4} & c \in (\frac{R}{4}, \frac{R}{2}], \\
    \frac{20}{R^3}u^4 - \frac{28}{R^2}u^3 + \frac{12}{R}u^2 + \frac{R}{4} & u = c - \frac{R}{2},\ c \in (\frac{R}{2}, R], \\
    c & c > R.
    \end{cases}
    """)

    # ################################################################ 
    # Plot des fonctions pour R(x) constant et exponentiel
    st.subheader("Variation de l'auto-évaluation et de l'apprentissage réel en fonction du temps")
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))

    # Objectif constant
    R_constant_values = R0 * np.ones_like(x)
    f_constant_values = f(x, R0, 0, f0, beta)
    axs[0].plot(x, R_constant_values, label=r'$R(t)$', color='red', linestyle='--')
    axs[0].plot(x, f_constant_values, label=r'$f(t)$', color='green')
    axs[0].plot(x, h(x, R0, 0, f0, beta), label=r'$h(t)$', color='blue')
    axs[0].set_title('Niveau d\'apprentissage avec objectif constant')
    axs[0].set_xlabel('Temps $t$')
    axs[0].set_ylabel('Niveau d\'apprentissage')
    axs[0].set_xlim(x_range[0], x_range[1])
    axs[0].set_ylim(y_range[0], y_range[1])
    axs[0].legend()
    axs[0].grid(True)

    # Objectif exponentiel
    R_exponential_values = Ref(x, R0, k)
    f_exponential_values = f(x, R0, k, f0, beta)
    axs[1].plot(x, R_exponential_values, label=r'$R(t)$', color='red', linestyle='--')
    axs[1].plot(x, f_exponential_values, label=r'$f(t)$', color='green')
    axs[1].plot(x, h(x, R0, k, f0, beta), label=r'$h(t)$', color='blue')
    axs[1].set_title('Niveau d\'apprentissage avec objectif croissant exponentiellement')
    axs[1].set_xlabel('Temps $t$')
    axs[1].set_ylabel('Niveau d\'apprentissage')
    axs[1].set_xlim(x_range[0], x_range[1])
    axs[1].set_ylim(y_range[0], y_range[1])
    axs[1].legend()
    axs[1].grid(True)

    st.pyplot(fig, use_container_width=True)

    # Affichage des équations
    # st.subheader("Équations des fonctions")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ref = objectif exponentiel en fonction du temps**")
        st.latex(r"""
        Ref(t,k) = R_0 \left( e^{k t} - k t \right)
        """)
        st.markdown(
            "où :\n"
            "- $R_0$ est le niveau de référence initial\n"
            "- $k$ est le taux de croissance exponentielle\n"
            "- $t$ est le temps"
        )
        
    with col2:

        
        st.markdown("**f = apprentissage au fil du temps**")
        st.latex(r'''
        f(t, R_0, k, f_0, \beta) = Ref(t, R_0, k) - (R_0 - f_0) \cdot t^{-\beta}
        ''')
        
        st.markdown("**h = evalearn(f(t)) = auto-évaluation au fil du temps**")
        st.latex(r"""
        h(t, R_0, k, f_0, \beta) = \mathrm{evalearn}\left( Ref(t,R_0,k) - (R_0 - f_0)t^{-\beta},\ Ref(t, R_0, k) \right)
        """)


    # ################################################################ 
    # Échantillons pour les axes
    t_vals = np.linspace(x_range[0], x_range[1], 500) #temps
    c_vals = np.linspace(y_range[0], y_range[1], 500) #niveau d'apprentissage
    # 1. Courbe evalearn: dans le plan (c, e)
    c_evalearn = c_vals
    e_evalearn = evalearn(c_evalearn, R0)
    trace_evalearn = go.Scatter3d(
        x=np.zeros_like(c_evalearn),  # t=0 pour ce plan
        y=c_evalearn,
        z=e_evalearn,
        mode='lines',
        name='evalearn (c, e)',
        line=dict(color='orange', width=4)
    )
    
    # 2. Courbe f: dans le plan (t, c)
    t_f = t_vals
    c_f = f(t_f, R0, k, f0, beta)
    trace_f = go.Scatter3d(
        x=t_f,
        y=c_f,
        z=np.zeros_like(t_f),  # e=0 pour ce plan
        mode='lines',
        name='f (t, c)',
        line=dict(color='yellow', width=4)
    )
    
    # Calcul Ref(t, R0, k)
    ref_f = Ref(t_f, R0, k)
    trace_fref = go.Scatter3d(
        x=t_f,
        y=ref_f,
        z=np.zeros_like(t_f),
        mode='lines',
        name='Ref (t, c)',
        line=dict(color='red', width=4, dash='dot')
    )
    
    # 3. Courbe h: dans le plan (t, e)
    t_h = t_vals
    e_h = h(t_h, R0, k, f0, beta)
    trace_h = go.Scatter3d(
        x=t_h,
        y=np.zeros_like(t_h),  # c=0 pour ce plan
        z=e_h,
        mode='lines',
        name='h (t, e)',
        line=dict(color='blue', width=4)
    )
    
    trace_href = go.Scatter3d(
        x=t_h,
        y=np.zeros_like(t_h),  # c=0 pour ce plan
        z=ref_f,
        mode='lines',
        name='Ref (t, e)',
        line=dict(color='red', width=4, dash='dot')
    )
    
    # 3. Courbe h: dans le plan (t, e)
    t_h = t_vals
    e_h = h(t_h, R0, k, f0, beta)
    trace_fh = go.Scatter3d(
        x=t_h,
        y=c_f,  # c=e pour le plan incliné
        z=e_h,
        mode='lines',
        name='h (t)',
        line=dict(color='green', width=6)
    )
    
    trace_fhref = go.Scatter3d(
        x=t_h,
        y=ref_f,  # c=e pour le plan incliné
        z=ref_f,
        mode='lines',
        name='Ref (t)',
        line=dict(color='red', width=6, dash='dot')
    )
    
    
    
    # Définition des bornes 
    t_min, t_max = t_vals[0], t_vals[-1]
    c_min, c_max = y_range[0], y_range[1]
    
    # Création de la grille pour le plan e=c
    t_plane = np.array([t_min, t_max, t_max, t_min])
    c_plane = np.array([c_min, c_min, c_max, c_max])
    e_plane = c_plane  # puisque e = c
    
    plane_trace = go.Mesh3d(
        x=t_plane,
        y=c_plane,
        z=e_plane,
        color='gray',
        opacity=0.2,
        name='Plan e=c',
        showscale=False
    )
    
    layout = go.Layout(
        scene=dict(
            xaxis_title='Temps (t)',
            yaxis_title='Compétence réelle (c)',
            zaxis_title='Compétence auto-évaluée (e)',
        ),
        title='Courbes 3D de compétence et auto-évaluation',
        height=900 
    )
    
    fig = go.Figure(data=[trace_evalearn, trace_f, trace_fref, trace_h, trace_href, plane_trace, trace_fh, trace_fhref], layout=layout)
    
    # Pour afficher avec Streamlit
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.title("Information")

    st.sidebar.image("static/images/logo_lifelonglearning_v3.png", use_container_width =True)
    st.markdown(get_readme_content())

