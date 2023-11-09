# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 09:52:35 2023

@author: ben69
"""

# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
from scipy.stats import chi2_contingency
from itertools import combinations
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# Bar sur le coté

# Insérer du code HTML pour personnaliser le style de la sidebar
sidebar_custom_style = """
<style>
[data-testid="stSidebar"] {
    background-color: lightblue;
}
.stOptionMenu {
    border-radius: 20px;
    padding: 20px;
    background-color: white;
}
.style{
    color: darkblue;
}
"""

# Afficher le style personnalisé dans la sidebar
st.markdown(sidebar_custom_style, unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title='Bank Marketing',
        options=['Introduction', 'Présentation des Données', 'DataViz', 'Transformation des Données', 'Notre Modèle', 'Conclusion', 'Démonstration','Faites votre Modélisation', 'Pour aller plus loin', 'Remerciements'])

# Navigation dans les options
if selected == 'Introduction':
    
    col1, col2, col3 = st.columns([1,25,1])
    with col1:st.write(" ")
    with col2:st.image('image_bank_streamlit.jpg', width=600)
    with col3:st.write(" ")
    
    st.markdown('<h3 style="color:darkblue;font-weight:bold;font-size:60px;">Projet Bank Marketing</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:black;font-weight:bold;font-size:25px;">Auteur:</h3>', unsafe_allow_html=True)
    st.markdown('Sarah, Balde, Thibault et Benjamin')
    st.markdown('<h3 style="color:black;font-weight:bold;font-size:25px;">Projet fil rouge de notre Formation DATA ANALYST</h3>', unsafe_allow_html=True)
    st.write('Formateur - DATASCIENTEST')
    
    st.write("---------------------------------------------------------------------------")
    
    st.markdown('<h3 style="color:black;font-weight:bold;font-size:30px;">Définition et Déroulement:</h3>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Définition:</h3>', unsafe_allow_html=True)
    st.write("Notre client, une banque, nous demande de **prédire la souscription ou non de ses clients à un produit financier** appelé, dépôt à terme. Pour cette demande, la banque nous fournit un jeu de données regroupant les informations des clients télé marketés.")
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Déroulement:</h3>', unsafe_allow_html=True)
    st.write("Pour répondre à cette question, nous étudierons dans un premier temps, les différentes caractéristiques du jeu de données. Dans un second temps, nous effectuerons **une analyse visuelle et statistique** des variables explicatives et de la variable cible. Ensuite, nous réaliserons des modèles de prévision avec les techniques de **Machine Learning**. Enfin, nous utiliserons les techniques d’**interprétation des modèles** afin d’expliquer pourquoi un client est plus susceptible de souscrire ou non à un dépôt à terme.")
    
    st.sidebar.write('------------------')
    st.sidebar.write('Auteurs:')
    st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
    st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
    st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
    st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
    st.sidebar.write('------------------')
    st.sidebar.image("datascientest_5fe203c15886e.jpg")
    st.sidebar.write("Formation Data Analyst - Janvier 2023")    
    
########################################################################
elif selected == 'Présentation des Données':
        
    option_submenu = st.sidebar.selectbox('Menu', ('Le Dataset', 'Nettoyage des Données'))
    if option_submenu == 'Le Dataset':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023")  
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Présentation des Données:</h3>', unsafe_allow_html=True)
        st.write("-----------------------------------------------------------------")
        st.subheader('Les Données')
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        st.dataframe(df.head(10))
    
        st.write('Dans notre jeu de données, **nous avons 45 211 lignes qui correspondent chacune à un client** contacté par la banque. On note **16 colonnes** en tant que variables explicatives et une variable "cible". Les variables explicatives sont utilisées dans le but d’expliquer ou de prédire la variable cible.')
        st.write("La **variable cible** est la variable dont on cherche à prédire la valeur à l’aide des variables explicatives. Dans cette analyse, nous disposons d’une seule variable cible appelée « deposit ». On cherche à prédire si un client va souscrire ou non à un dépôt à terme.")
        st.write('-------------------------------------------------------------------------')
    
    #Présentation des Variables
        st.subheader('Présentation des Variables')
        option = st.selectbox(
            'Choisissez une colonne',
            df.columns)
        
        st.write(f'### Type de Donnée')
        st.write('les données de cette variable sont de type:', df[option].dtypes)
    
        st.write(f'### Informations')
        st.write(df[option].describe())
    
        st.write(f'### Valeurs Uniques')
        st.write(df[option].unique())
        
        st.write("Vous trouverez le **type de la donnée** :green[en premier], **les informations importantes de la variable choisie**,:green[en second] et :green[enfin], vous trouverez les **valeurs uniques.**")
        
    # Valeurs Importantes 
    
    elif option_submenu == 'Nettoyage des Données':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Présentation des Données:</h3>', unsafe_allow_html=True)
        st.subheader('Nettoyage des Données')
        st.subheader('Définition du Process')
        st.write("Comme leurs noms l'indique, les manquants et les doublons sont très importants à étudier. En effet, si notre jeu de données en présente, les résultats pourraient s'en trouver biaisés.")
        st.write("Il est donc nécessaire de vérifier la présence ou non de manquants et de doublons dans notre jeu de données.")
      # Manquants  
        st.markdown('-------------------------------------------')
        st.subheader('Manquants et Doublons')
        st.markdown(f'### Manquants')
        st.markdown('Regardons les manquants et les doublons parmi nos colonnes.')
        st.markdown("Nous vous laissons vérifier par vous même la présence de manquants et doublons mais aucun n'est a déplorer dans notre jeu de données.")
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        option = st.selectbox(
            'Choisissez une colonne',
            df.columns)
        
        st.write(f'### Manquants')
        st.write(f'## Le nombre de manquants est de:', df[option].isna().sum())
        
      # Doublons  
        st.markdown('-------------------------------------------')
        st.markdown(f'### Doublons')
        st.write(f'## Le nombre de doublons est de :', df.duplicated().sum())
        st.markdown("Vous l'aurez compris, le nettoyage de notre jeu de données est assez simple et rapide puisque nos données 'brutes' sont déjà bien fournies et renseignées.")
########################################################################
elif selected == 'DataViz':
    option_submenu = st.sidebar.selectbox('Menu Data Visualisation', ('Cible', 'Corrélation', 'Les Variables Explicatives', 'Décisions', 'Faites votre Dataviz'))
    if option_submenu == 'Cible':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        #Plotly de la Variable cible
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Dataviz</h3>', unsafe_allow_html=True)
        st.write(":red[Information de Navigation.]")
        st.write("Afin de naviguer entre les sous-menus, nous vous invitons à selectionner la rubrique que vous souhaitez dans le menu de gauche: Menu Data Visualisation.")
        st.write("----------------------------------------------------------------------")
        st.subheader('La Variable Cible')
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        palette_color = ['darkcyan', 'tomato']
        sns.set(rc={'axes.facecolor':'lightgray'})
        
        cible = px.histogram(df, x='deposit', title='Distribution de la Variable Deposit', 
                             color='deposit', 
                             labels={'deposit': 'Souscription'}, 
                             color_discrete_sequence=['darkcyan', 'tomato'])
        
        st.plotly_chart(cible)
        
        st.markdown('Cet histogramme représente **la distribution de notre variable cible**. En effet, nous pouvons facilement remarquer que dans notre jeu de données, les clients **n’ayant pas souscrit** au dépôt à terme sont majoritaires et **représentent 88,3% des valeurs** contre 11,7% pour les clients ayant souscrit.')
        st.markdown('Ce graphique est important pour une autre raison : étant donné que seulement 11,7% des clients ont souscrit à un dépôt à terme, notre jeu de données peut être qualifié de **déséquilibré**, et par conséquent, un modèle de prédiction appliqué directement sur ce jeu de données serait considéré comme « naïf » puisqu’il obtiendrait un score de 88,3 % de bonnes prédictions pour les personnes n’ayant pas souscrit. Finalement, un **rééchantillonnage sera très certainement nécessaire** avant l’étape de prédiction afin d’avoir des résultats plus cohérents.')
        
    elif option_submenu == 'Corrélation':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Dataviz</h3>', unsafe_allow_html=True)
        st.subheader('Corrélation entre nos Variables Numériques')
        
    #Matrice de Corrélation
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        df_num = df.select_dtypes(include='int64')
        df_num['deposit'] = df['deposit'].replace({'no':0, 'yes':1})
        fig, ax = plt.subplots(figsize=(8,8))
        sns.heatmap(df_num.corr(), annot=True, cmap='viridis')
        plt.title('Matrice de Corrélations');
        st.pyplot(fig)
        st.markdown('On remarque que toutes les variables numériques de notre jeu de données sont représentées sur l’axe des abscisses et des ordonnées. Le point d’intersection de deux variables correspond à **la valeur de la corrélation entre ces variables** allant de -1 (variables négativement corrélées) jusqu’à 1 (variables parfaitement corrélées). En légende à droite, on retrouve une échelle de couleurs indiquant le degré d’intensité des corrélations entre les variables. En considérant uniquement les corrélations significatives (p-value < 0.05) avec un coefficient (r) ≥ 0.3, nous pouvons observer **seulement deux corrélations d’intensité modérée** entre : « pdays » et « previous » (r = 0.45), ainsi qu’entre « deposit » et « duration » (r = 0.39).')
        
        st.write("-----------------------------------------------------------")
        st.subheader('Corrélation entre nos Variables Catégorielles')
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        df_cat = df.select_dtypes(include='object')
        col1 = st.selectbox("Sélectionner la première colonne:", df_cat.columns)
        col2 = st.selectbox("Sélectionner la seconde colonne:", df_cat.columns)
        
        from scipy.stats import chi2_contingency
        from itertools import combinations
        import numpy as np
        
        st.write("Corrélation entre",col1, "et", col2)
        N = df_cat.shape[0]
        table = pd.crosstab(df_cat[col1],df_cat[col2])
        stat_chi2 = chi2_contingency(table)
        k = table.shape[0]
        r = table.shape[1]
        phi = max(0,(stat_chi2[0]/N)-((k-1)*(r-1)/(N-1)))
        k_corr = k - (np.square(k-1)/(N-1))
        r_corr = r - (np.square(r-1)/(N-1))
        vcram = np.sqrt(phi/min(k_corr - 1,r_corr - 1))
        st.write('V de Cramer / p-value:', vcram, '/', stat_chi2[1],"\n")
        if vcram > 0.5:
            st.write("La valeur du V-Cramer indique qu'il y à une forte corrélation entre les variables étudiées.")
        elif vcram < 0.2:
            st.write("La valeur du V-Cramer indique qu'il y à une faible corrélation entre les variables étudiées.")
        else:
            st.write("La valeur du V-Cramer indique que la corrélation entre les variables étudiées n'est pas négligeable.")
        
        st.write("-----------------------------------------------------------")
        st.subheader('Corrélation entre nos Variables Catégorielles et Numériques')
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        df_cat = df.select_dtypes(include='object')
        df_num = df.select_dtypes(include=['float64', 'int64'])
        col1 = st.selectbox("Sélectionner la variable numérique:", df_num.columns)
        col2 = st.selectbox("Sélectionner la variable catégorielle:", df_cat.columns)
        st.write("Corrélation entre", col1, "et", col2)
        import statsmodels.api
        result = statsmodels.formula.api.ols(col1 + '~' + col2, data = df).fit()
        table = statsmodels.api.stats.anova_lm(result)
        p_value = table['PR(>F)'][col2]
        st.write("La P_Value est de:", p_value)
        st.write("Si la valeur de la p-value est inférieur à 5%, nous rejettons l'hypothèse selon laquelle la variable", col2, "n'a pas d'influence sur la variable",col1,".")
        
#####################################################################################################################################
    elif option_submenu == 'Les Variables Explicatives':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Dataviz</h3>', unsafe_allow_html=True)
        st.subheader('Les Variables Catégorielles')
        st.markdown("Nous vous proposons une **visualisation des différentes variables catégorielles en fonction de la variable cible** de notre jeu de données.")
        
    # Graphiques des variables catégorielles
        palette_color = ['darkcyan', 'tomato']
        sns.set(rc={'axes.facecolor':'lightgray'})
        
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        df_cat = df.select_dtypes(include='object')
        exclusion = ['deposit']
        colonnes = [col for col in df_cat.columns if col not in exclusion]
        option = st.selectbox('Sélectionner une colonne:', colonnes)
        fig, ax = plt.subplots(figsize=(20,12))
        fig = px.histogram(df_cat, x=[option], title=f'Distribution de la Variable {option} vs Deposit', 
                             color='deposit', 
                             labels={'deposit': 'Souscription'}, 
                             color_discrete_sequence=['darkcyan', 'tomato'],
                             barmode='group')
        st.plotly_chart(fig)
        
        if option == "job":
            st.write("Les « managers », « techniciens », « administrateurs » ainsi que les « blue-collar » souscrivent plus à un dépôt à terme que les clients ayant un autre métier. Les « blue-collar » représentent les emplois manuels, souvent associés à l'industrie manufacturière, la construction, la mécanique automobile et d'autres métiers similaires. Les retraités ne sont pas beaucoup représentés dans notre jeu de données, mais il est important de noter que près de la moitié de clients retraités ont souscrit à l'offre.")
        elif option == "marital":
            st.write("Nous avons beaucoup de personnes mariées qui souscrivent à l'offre et également beaucoup qui la refusent. La classe célibataire semble avoir le taux de souscription à l'offre le plus élevé. Pour les divorcés il y a une sous-représentation de la classe mais le nombre de personnes qui ont souscrit par rapport à ceux qui ont participé n'est pas négligeable. Les personnes qui vivent seules sont plus enclines à souscrire à l'offre.")
        elif option == "education":
            st.write("Les personnes avec un niveau d'éducation secondaire ont le taux de souscription le plus élevé. On peut imaginer que plus le niveau d'éducation est élevé, plus les personnes ont tendance à souscrire à un dépôt à terme.")
        elif option == "default":
            st.write("Bien que la quasi-totalité (98.5%) des clients recensés n'ont pas eu de défaut de paiement (incident bancaire lié au remboursement d’un prêt, compte à découvert etc.), on remarque qu'il y a à peu près deux fois moins de clients (1% vs 1.8%) avec un historique de défaut de paiement qui ont accepté de souscrire à un dépôt à terme par rapport à ceux qui ne sont pas en défaut.")
        elif option == "housing":
            st.write("C'est une variable qui peut s’avérer informative car elle permet de voir que parmi les clients qui ont accepté de souscrire au dépôt à terme, il y a environ 36.8% de propriétaires contre 63.2% de locataires. Le dépôt intéresserait donc potentiellement plus les locataires, qui sont peut-être plus enclins à épargner du fait qu’ils n’ont à priori pas investi dans l’achat d’une résidence principale.")
        elif option == "loan":
            st.write("Globalement, la très grande majorité des clients (87%) n'ont pas contracté, préalablement, un autre prêt auprès de la banque ; cependant on remarque qu'il y a presque deux fois plus de clients avec un prêt préexistant (9.3% vs 17%) qui n'ont pas accepté de souscrire à ce produit par rapport aux clients qui n'avaient aucun prêt en vigueur à la banque.")
        elif option == "contact":
            st.write("Ce graphique met en évidence le type de moyen de communication que la campagne télémarketing a utilisé pour contacter le client de la banque. On voit en orange les personnes qui ont contracté un prêt suite à cette campagne et en bleu celles qui n’ont pas contracté de prêts.")
        elif option == "month":
            st.write("Ce graphique met en évidence le mois où la campagne télémarketing contacte le client de la banque. On voit en orange les personnes qui ont contracté un prêt suite à cette campagne et en bleu celles qui n’ont pas contracté de prêts. Les mois de Mai (13 766), Juillet et Août, on remarque que les clients sont plus souvent contactés mais ce n'est pas pour autant qu'ils contractent un dépôt de prêt.  Le mois de décembre est le mois où la campagne télémarketing a contacté le moins de clients (214).")
        elif option == "poutcome":
            st.write("La variable « Poutcome » indique si le client a souscrit au dépôt à terme lors de la campagne marketing précédente. Cette variable n’est pas la plus intéressante, en revanche on note une grande majorité de valeurs dans la classe « unknown ». Nous proposons de regrouper les valeurs « unknown » et « other » dans la même catégorie afin que cela soit le plus simple possible pour notre futur modèle. Cette décision est également motivée par le fait que la catégorie « other », représente seulement 4,07% du jeu de données.")
        elif option == "deposit":
            st.write("Concernant la variable cible, deposit, nous vous proposons de vous reporter au sous-menu 'Variable Cible'.")
        
        
        
        
        st.markdown('--------------------------------------------------------------')
        st.subheader('Les Variables Numériques')
        st.markdown("Nous vous proposons le même type d'animation concernant les variables numériques. Veuillez noter que pour ce graphique, **les valeurs en ordonnées représentent la densité probable**.")
        
    # Graphiques des variables numériques
        palette_color = ['darkcyan', 'tomato']
        sns.set(rc={'axes.facecolor':'lightgray'})
       
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        df['deposit'] = df['deposit'].replace({"no":0, "yes":1})
        df_num = df.select_dtypes(include='int64')
        exclusion_num = ['deposit']
        colonnes_num = [col for col in df_num.columns if col not in exclusion_num]
        numerique = st.selectbox('Sélectionner une colonne', colonnes_num)
        fig2, ax = plt.subplots(figsize=(20,12))
        fig2 = px.histogram(df_num, x=[numerique], title=f'Distribution de la Variable {numerique} vs Deposit', 
                             color='deposit',
                             labels={'deposit': 'Souscription'}, 
                             color_discrete_sequence=['darkcyan', 'tomato'],
                             histnorm='probability density',
                             barmode='group')
        st.plotly_chart(fig2)

        if numerique == "age":
            st.write("On remarque que la majorité des clients ont entre 30 et 60 ans. Les âges entre 30 et 35 ans sont les plus représentés sur ce graphique. On remarque que les deux courbes se rejoignent après 60 ans c’est-à-dire qu’il y a autant de personnes qui acceptent l’offre que de personnes qui la refusent. Nous avons décidé de regrouper plusieurs valeurs de cette variable entre elles : « - de 20 ans », « entre 20 et 35 ans », « entre 35 et 50 ans », « entre 50 et 65 ans », « entre 65 et 80 ans » et enfin « + de 80 ans », afin de tenter de mieux caractériser d’éventuels liens entre ces tranches d’âge et notre variable cible.")
        elif numerique == "balance":
            st.write("La distribution des valeurs de la variable « balance » est particulièrement étendue. Comme expliqué précédemment, elle représente le compte bancaire des clients dont le solde s’étale entre -8 019€ à 102 127€.") 
            st.write("Cependant, nous avons pu constater qu’environ 98% des valeurs de cette variable sont comprises entre -1 000€ et 10 000€. Il paraît donc justifié de recentrer les valeurs de cette variable au sein de ces deux bornes (-1 000€ et 10 000€) afin d’éliminer les valeurs les plus extrêmes et d’obtenir ainsi une distribution plus pertinente (telle que représentée dans la figure ci-dessous), tout en s’efforçant de perdre le moins de données possibles.")
        elif numerique == "day":
            st.write("Ce graphique représente le jour du mois auquel le client a été contacté par la campagne de télémarketing. Comme au-dessus, la couleur orange représente les clients qui ont à la suite de cet appel, souscrit à un prêt et en bleu les autres. On n’observe pas de tendance nette. La distribution de la variable explicative semble uniforme avec des pics et des creux dus, je pense, au hasard du calendrier.")
        elif numerique == "duration":
            st.write("Ce graphique met en évidence le temps en seconde de la durée de l’appel de la campagne télémarketing. On voit en orange les personnes qui ont contracté un prêt suite à cette campagne et en bleu celles qui n’ont pas contracté de prêts. La durée maximum d’un appel téléphonique est de 4 918 secondes.")
        elif numerique == 'campaign':
            st.write("Pour la variable « campaign », nous remarquons que la majorité des clients ont été appelés entre zéro et cinq fois.")
        elif numerique == "previous":
            st.write("Cette variable représente le nombre d’appels vers le client par la banque pour la dernière campagne marketing.")
        elif numerique == "pdays":
            st.write("Concernant la variable « pdays », elle représente le nombre de jours depuis le dernier appel de la banque à notre client. Nous remarquons que la plage de données est extrêmement longue.")







#####################################################################################################################################
    elif option_submenu == 'Décisions':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023")
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Dataviz</h3>', unsafe_allow_html=True)
        st.subheader("Décisions avant l'étape de la modélisation")
        
        st.markdown('L’étape de data visualisation nous a permis de **mieux caractériser les valeurs et l’étendue de nos données**, et de prendre également d’importantes décisions quant au **regroupement possible de variables et aux réductions de nos plages de données**.')
        st.markdown('Nous vous proposons **la liste des changements opérés sur notre jeu de données** avant la phase de modélisation :')
        st.markdown('-	La variable « Age » va être regroupée en plusieurs groupes afin de catégoriser les clients en fonction de leur âge. Les groupes seront [-20ans], [20-35ans], [35-50ans], [50-65ans], [65-80ans] et [+80ans].')
        st.markdown('-	La variable « Balance » va être réduite en les valeurs [-1000 et 10000]. Nous parlons ici de la somme sur le compte en banque de notre client.')
        st.markdown('-	La variable « Duration » va être réduite en les valeurs [0 et 600]. Nous parlons ici de secondes.')
        st.markdown('-	La variable « Pdays » va être réduite en les valeurs [-1 et 400]. Nous parlons ici du nombre de jours depuis le dernier appel de la banque à notre client. Nous allons également rapprocher les valeurs [-1] et les valeurs [0] sous la même valeur [0].')
        st.markdown('-	La variable « Campaign » va être réduite en les valeurs [0 et 20]. Nous parlons ici du nombre d’appels que le client a reçu lors de la campagne marketing actuelle.')
        st.markdown('-	La variable « Previous » va être réduite en les valeurs [0 et 10]. Nous parlons ici du nombre d’appels que le client a reçu lors de la campagne marketing précédente.')
        st.markdown('-	La variable « Poutcome » : concernant cette variable, nous allons fusionner les valeurs « other » et « unknown » sous la même valeur « other ».')
        
        st.title('Nouveau jeu de Données:')
        df = pd.read_csv('Dataset-apres-DataVisualisation')
        df = df.drop('Unnamed: 0', axis=1)
        st.dataframe(df)
        
#####################################################################################################################################
    elif option_submenu == 'Faites votre Dataviz':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Dataviz</h3>', unsafe_allow_html=True)
        st.subheader("Faites votre Dataviz")
        
        df = pd.read_csv("Copie de bank-full.csv", sep=';')
        variables_numeriques = df.select_dtypes(include=['int64', 'float64'])
        variables_categorielles = df.select_dtypes(include='object')
        
        st.subheader('Visualisation des variables Numériques')
        abscisses = st.selectbox('Sélectionnez une valeur en Abscisses:', variables_numeriques.columns)
        ordonnees = st.selectbox('Sélectionner une valeur en Ordonnées:', variables_numeriques.columns)
        
        fig = px.scatter(df,
                         x=abscisses,
                         y=ordonnees,
                         color=ordonnees,
                         title=(str(abscisses) + " vs " + str(ordonnees)))
        st.plotly_chart(fig)
        
        st.write('--------------------------------------------------------------------')
        
        st.subheader('Visualisation des Variables Catégorielles')
        abscisses_graphique_2 = st.selectbox('Sélectionner une valeur en Abscisses:', variables_categorielles.columns)
        
        fig = px.histogram(df,
                           x=abscisses_graphique_2,
                           color_discrete_sequence=['darkcyan'],
                           title=('Distribution de ' + str(abscisses_graphique_2)))
        
        st.plotly_chart(fig)
        
        st.write("-----------------------------------------------------------------------")
        st.subheader("Visualisation d'une Variable Catégorielle en fonction d'une Variable Numérique.")
        abscisses_2 = st.selectbox("Veuillez sélectionner la variable catégorielle:", variables_categorielles.columns)
        ordonnees_2 = st.selectbox("Veuillez sélectionner la variable numérique:", variables_numeriques.columns)
        nom_abscisse = abscisses_2
        nom_ordonnee = ordonnees_2
        fig = px.box(df,
                     x=abscisses_2,
                     y=ordonnees_2,
                     color_discrete_sequence=['tomato'],
                     title=("Représentation de" +' '+ str(ordonnees_2) +' '+ "en fonction de"+' ' + str(abscisses_2)),
                     labels={abscisses_2: nom_abscisse, ordonnees_2: nom_ordonnee})
        st.plotly_chart(fig)
        st.write("Nous vous proposons une représentation par **boxplot**, ce qui nous permet d'afficher les informations importantes pour **chaque itération en fonction de la variable numérique choisie**.")
        st.write("Les informations importantes sont : le **minimum**, le **maximum**, la **médiane**, la **valeur inférieur et la valeur supérieur avant les valeurs extrêmes** et enfin, les deux **délimitations des box.**")
        
#############################################################################################################

elif selected == 'Transformation des Données':
    st.sidebar.write('------------------')
    st.sidebar.write('Auteurs:')
    st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
    st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
    st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
    st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
    st.sidebar.write('------------------')
    st.sidebar.image("datascientest_5fe203c15886e.jpg")
    st.sidebar.write("Formation Data Analyst - Janvier 2023") 
    st.sidebar.write('------------------')
    st.sidebar.image('image_bank_streamlit.jpg')
        
    st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Transformation des données</h3>', unsafe_allow_html=True)
    st.write("Transformer les variables numériques et catégorielles **est primordial** pour que nos futurs modèles puissent travailler correctement. Les transformations peuvent être multiples : standardisation, encodage, traitement des manquants. Dans le cadre de notre étude, nous allons **encoder les variables catégorielles et nous allons standardiser nos variables numériques**.")
        
    st.subheader("Les Variables Catégorielles")
    st.write("Commençons par les variables catégorielles ! ")
    st.write("Voici une image de notre jeu de données regroupant les variables catégorielles:")
    st.image('X_train_cat.jpg')
      
    st.write("Voici maintenant notre jeu de données, mais cette fois, encodé, c'est à dire que nous avons **transformé nos données pour qu'elles soient toutes de type numérique**, en affectant une valeur à chaque itération des variables catégorielles.")
    st.image("X_train_cat_encoded.jpg")
    st.write("Pour cet encodage, nous avons utilisé l'encoder ONE HOT.")
    st.image("onehotencoder.jpg")
    st.write("Depuis le jeu de données original de gauche, le one hot encodeur a créé 3 nouvelles colonnes représentant de façon binaire chaque fruit. Nous allons appliquer la même méthode sur notre jeu de données afin de permettre à nos futurs modèles d’interpréter ce type de données car certains algorithmes de Machine Learning ne sont tout simplement pas en mesure de traiter des données catégorielles. Nous avons utilisé la même méthode sur notre jeu de données.")
    
    st.write("---------------------------------------------------------------")    
    st.subheader("Les Variables Numériques")   
    st.write("Nous allons appliquer **une standardisation sur nos variables numériques**. Elle va permettre de centrer et réduire la distribution de nos données afin que la moyenne des valeurs observées soit de 0 et l’écart type de 1. Cela aura pour effet de supprimer le biais dû à la présence de variables ayant une échelle de valeurs très différentes entre elles.")
    st.write("Voici les variables numériques de notre jeu d'entraînement avant la standardisation:")
    st.image("X_train_scaler.jpg")

    st.subheader("Nos variables explicatives sont maintenant encodées et standardisées.")
#############################################################################################################    
elif selected == 'Notre Modèle':
    option_submenu = st.sidebar.selectbox('Menu', ('Création du modèle', 'Notre Modèle', 'Interprétation'))
    if option_submenu == 'Création du modèle':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Notre modèle</h3>', unsafe_allow_html=True)
        st.write(":red[Information de Navigation.]")
        st.write("Afin de naviguer entre les sous-menus, nous vous invitons à selectionner la rubrique que vous souhaitez dans le menu de gauche: Menu Data Visualisation.")
        
        st.write("----------------------------------------------------------------------")
        st.header('Présentation du concept de modélisation')
        st.write("Le Machine Learning consiste à **repérer les répétitions dans un jeu de données** afin de les analyser et in fine pouvoir prédire un résultat. Il existe deux grands ensembles de Machine Learning : les systèmes **non supervisés et les supervisés**. Un système **supervisé** est un type de modèle d'apprentissage **utilisant un ensemble de données étiquetées pour apprendre à prédire ou à classer de nouvelles données**. Dans un système supervisé, l'algorithme reçoit un ensemble d'exemples de données avec leurs étiquettes correspondantes, et il apprend à faire des prédictions en se basant sur ces exemples. Un système non supervisé est un type de modèle d'apprentissage automatique qui **explore un ensemble de données non étiquetées**. Les systèmes non supervisés n'ont pas de données d'entraînement étiquetées. L'algorithme doit donc trouver par lui-même des structures ou des modèles intéressants dans les données.")
        st.write("Dans notre étude de données, nous sommes dans le cadre d’un système supervisé. Nous allons chercher à prédire la valeur cible c’est-à-dire à prédire si le client souscrit ou non à un dépôt à terme à la suite d’une campagne marketing. ")
        
        st.write("----------------------------------------------------------------------")
        st.subheader("Plusieurs étapes sont nécessaires à la réalisation d'une modèlisation:")
        st.write("**Etape 1 :** Séparer notre jeu de données en un jeu de données d'entraînement et un jeu de données de test.")
        st.write("- Jeux d'entrainement: 80% et jeu test: 20%")
        st.write("**Etape 2 :** Appliquer à ces jeux de données les transformations nécessaires : encodage des valeurs catégorielles et standardisation des variables numériques. Cette étape a été détaillée dans le menu 'Transformation des Données'.")
        st.write("**Etape 3** : Entraînement de nos modèles sur le jeu d'entraînement et prédiction sur le jeu de test.")
        st.write("- SVM")
        st.write("- K-Neighbors")
        st.write("- DecisionTree")
        st.write("- RandomForest")
        st.write("- LogisticRegression")
        st.write("**Etape 4 :** Analyse des résultats de nos modèles.")
        st.write("**Etape 5 :** Optimisation du meilleur modèle.")
        st.write("**Etape 6 :** Interprétation du meilleur modèle.")
        
        st.write("----------------------------------------------------------------------")
        st.header("Présentation de nos résultats")
        code=st.checkbox("Afficher le code utilisé, pour entrainer, prédire, mise en forme des dataframes de résultats et affichage des graphiques de résultats")
        if code:
            st.image("def predictions.png")
            st.image("Utilisation de la fonction prediction.png")
            st.image("Affichage du graphique.png")
            
        
        st.write("Veuillez cliquer sur le bouton afin d'afficher les résultats.")
        if st.button("Résultats sur le Dataset Original"):
            st.image("nouveau df resultats originaux avec accuracy train.jpg")
            st.image("resultats originaux.jpg")
            st.write("**Meilleur modèle:**")
            st.image("image.png")
        if st.button("Résultats avec utilisation des poids de classe"):
            st.image("nouveau df resultats balanced.jpg")
            st.image("resultats balanceeed.jpg")
        if st.button("Résultats après utilisation du Smote"):
            st.image("nouveau df resultats smote.jpg")
            st.image("resultats smotttt.jpg")
        
        st.write("----------------------------------------------------------------------------")
        st.subheader("Optimisation des hyper-paramètres")
        code_2 = st.checkbox("Afficher le code utilisé pour optimiser les hyper-paramètres des deux modèles sélectionnés")
        if code_2:
            st.image("gridsearchcv.png")
            st.image("entrainement gridsearchcv.png")
            st.image("graph gridsearchcv.png")
        st.write("-----------------------------------------------------------------------------")
        
        st.subheader("Résultats de l'optimisation des hyper-paramètres")
        st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Résultats originaux:</h3>', unsafe_allow_html=True)
        st.image("svc sans grid search.png")
        st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Resultats avec GridSearchCV:</h3>', unsafe_allow_html=True)
        st.image("svcgridsearch.png")
        
        
        st.write("-----------------------------------------------------------------------------")
        st.write("Nous avons donc entrainé et nous connaissons **les hyper-paramètres du meilleur modèle**. Il nous faut maintenant l'expliquer au plus grand nombre et surtout à notre client qui n'a pas forcément des compétences dans le domaine de la data.")
        

    if option_submenu == 'Notre Modèle':
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Notre modèle</h3>', unsafe_allow_html=True)
        st.write("Après avoir défini les étapes de la création de notre modèle nous allons donc vous présenter le modèle que nous avons choisi et par la suite, nous allons l'interpréter.")
        st.write("Notre modèle sera un modèle **SVM** (Sector Vector Machine).")
        st.write('L’idée du Support Vector Machine est de **projeter les données dans un espace de plus grande dimension**, pour les rendre séparables.')
        st.write("Voici les paramètres de notre modèle SVM:")
        st.write("- C = 5.0")
        st.write("- Kernal = RBF")
        st.write("- gamma = auto")
        st.write("- class_weight = balanced")
        st.write("- probability = True")
        st.write("---------------------------------------------------------")
        st.header("Résultats de notre modèle")
        if st.button("Matrice de Confusion"):
            st.write("Nous vous proposons dans un premier temps la matrice de confusion et le rapport de performances.")
            st.image('svc.png')
            st.image("svc umbalanced.png")
            st.write("La matrice de confusion est intéressante. En effet, on remarque que la classe « 0 » **est très majoritairement présente**, ce qui correspond parfaitement au fait que notre jeu de données initial est très déséquilibré. **Cette classe sera donc toujours beaucoup plus représentée** que la classe « 1 » (classe minoritaire).")
            st.write("On peut remarquer que le modèle SVC est un **modèle beaucoup plus intéressant** en termes de performance que les autres. Remarquez d'ailleurs que ce modèle est le plus performant pour minimiser les faux positifs. Ces valeurs représentant les clients souscrivant à notre produit mais qui sont mal prédits par notre modèle. Cette valeur **est particulièrement importante dans ce rapport** puisque nous ne voulons pas perdre de valeur représentant les clients ayant souscrit à notre produit, car ils sont minoritaires. En revanche, si notre modèle prédit qu'un client a des chances de souscrire alors qu'il ne va pas le faire cela est moins problématique. Autrement dit, la prédiction du nombre de faux négatifs est un facteur moins primordial.")
            st.write("Le rapport des performance de notre modèle nous indique plusieurs choses importantes. La première est la valeur de la **métrique 'geo' qui permet de visualiser le poid atrribué à chaque classe de notre jeu de données**. Un score de 86 est très bon sachant que nous avions un déséquilibre certain. Nous avons également deux métriques importantes qui sont les **recall et le f1-score**. Nous avons précédemment expliqué l'importance de ces deux valeurs. Notre modèle obtient pour la classe 0 et la classe 1 respectivement 0.83 et 0.89 de recall et 0.90 et 0.57 de f1-score. Nous pouvons être satisfait de ces résultats même s'ils peuvent toujours être optimisés.")
        
            st.image("roc svc.png")
            st.write("**Une courbe ROC (Receiver Operating Characteristic)** est un graphique qui montre les performances d'un modèle de classification à tous les seuils de classification.")
            st.write("**Plus la courbe approche le coin en haut à gauche, plus le modèle est performant**. A l'inverse un modèle approchant la courbe en pointillés bleu, serait un modèle qui réaliserai ses prédictions selon le principe d'une chance sur deux.")
            
        st.write("-----------------------------------------------------")     
        st.header("Reduction de Dimension") 
        st.image("pca cercle.jpg")
        st.write("Un point important afin d'optimiser les performances de notre modèle, serait d'étudier les variables importantes et de réduire le jeu de données. Nous avons réalisé une étude PCA que nous présentons ci-dessus. Elle ne s'est pas révélée pertinente. Effectivement 90% variance expliquée de l'information de notre jeu de données réside dans 13 composantes principales. Elle tombe à 10 composantes prinicpales pour expliquer 80% de l'information.")
    if option_submenu == 'Interprétation':
        
        st.sidebar.write('------------------')
        st.sidebar.write('Auteurs:')
        st.sidebar.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
        st.sidebar.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
        st.sidebar.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
        st.sidebar.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
        st.sidebar.write('------------------')
        st.sidebar.image("datascientest_5fe203c15886e.jpg")
        st.sidebar.write("Formation Data Analyst - Janvier 2023") 
        st.sidebar.write('------------------')
        st.sidebar.image('image_bank_streamlit.jpg')
        
        st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Notre modèle</h3>', unsafe_allow_html=True)
        st.subheader("Interprétation de notre modèle")
        st.write(" Il est **PRIMORDIAL** d'interpréter notre modèle. En effet, l'objectif est **d'expliquer le fonctionnement de notre modèle au plus grand nombre**. Autrement dit expliquer pourquoi le modèle a rendu telle ou telle prédiction.")
        st.write("Il existe plusieurs librairies nous permettant d'analyser le fonctionnement d'un modèle de machine leraning. Dans notre cas de figure, nous vous présenterons trois librairies qui sont :")
        st.write("- SKATER")
        st.write("- SHAP")
        st.write("- LIME")
        
        st.write("**Note importante:** Pour des raisons de fluidité de notre application, nous ne ferons pas tourner de lignes de code pour cette partie. Nous vous proposons les visualisations que nous avons utilisé dans notre rapport.")
        
        #Boutons
        if st.button("Interprétation avec SKATER"):
            st.subheader("Interprétation avec SKATER")
            st.subheader("Analyse des variables importantes")
            st.image("interpretation features importances.jpg")
            st.write("Sur ce graphique nous remarquons que la variable « duration » est, de loin, la variable la plus influente dans la prise de décision de notre modèle. Les variables « pdays », « day » et « previous » sont également importantes mais dans une bien moindre mesure comparée à la variable précédente. Également certaines itérations des variables « month » et « contact » sont importantes. ")
            st.write("A l'inverse, les variables « age » (représentant les clients de plus de 80 ans et de moins de 20 ans), « default », « job », et « education » représentent les variables les moins influentes dans le processus de prédiction de notre modèle.")
            st.write("Concernant les autres variables, nous remarquons que les variables « balance » et « campaign » sont assez similaires en termes d’importance et représente un impact non négligeable sur le modèle. À l’inverse, il est intéressant de noter que la variable « housing » n’est pas si pertinente dans le sens où les deux itérations de notre variable sont similaires en termes d’importance. Autrement dit, que notre client soit propriétaire ou non n’affecte pas la décision du modèle.") 
            st.write('------------------')
            st.subheader("Analyse variable par variable")
            st.image("duration interpretation.jpg")
            st.write("Afin d’interpréter ces graphiques, il est nécessaire de préciser quelques points importants. Un « PD plot » représente l’impact marginal d’une variable donnée sur la prédiction du modèle. En abscisse, vous retrouvez les valeurs de la variable étudiée, alors que l’ordonnée représente les valeurs de probabilités d'appartenir à la classe « 1 » (qui correspond à la classe des clients ayant souscrit au produit). La zone bleue qui englobe la courbe représente la variance des valeurs de probabilités pour une valeur de la variable (« duration »)'ensemble des valeurs des prédictions prises par notre modèle en fonction des valeurs de la durée d’appel.")
            st.write("Sans surprise et au regard de l’importance de la variable « duration » dans le processus de prédiction de notre modèle, il semble que plus la durée de l’appel est longue, plus le client a de chance de souscrire au produit que nous lui proposons.")
            st.write("----------------------------")
            st.image("day interpretation.jpg")
            st.write("Nous avons eu raison de ne pas formuler des conclusions trop hâtives sur cette variable lors de la première analyse. Ce graphique est intéressant car même si la variable « day » semble être importante pour notre modèle d’après la première analyse, en réalité, tous les jours de la semaine semblent être aussi importants les uns que les autres. Autrement dit, le client a autant de chance de souscrire au produit financier quel que soit le jour où il est appelé. Cette variable n’est pas pertinente pour la finalité de l’étude, puisqu'elle ne permet pas de mieux cibler les clients qui vont souscrire au dépôt.")
            st.write("----------------------------")
            st.image("pdays interpretation.jpg")
            st.write("La variable « pdays » est assez similaire à la variable « day » c’est-à-dire que quel que soit le nombre de jours entre les deux derniers appels effectués par la banque, le client a autant de chance de souscrire au produit financier. La très légère baisse dans les variations de prédiction intervenant à partir de la valeur 0.5 de « pdays » n’affecte que très peu les chances de souscription. Nous observons également, que ces dernières remontent à partir de la valeur 2.5.")
            st.write("----------------------------")
            st.image("campaign interpretation.jpg")
            st.write("Il semble que plus le client reçoit d’appels pendant la campagne marketing, moins il a de chance de souscrire au produit. C’est une indication intéressante, car manifestement, le client doit être appelé pendant la campagne mais il ne faut pas non plus le « harceler » car il pourrait ne plus être à réceptif aux informations que nous lui délivrons. Point très important à retenir et auquel on pouvait s’attendre, puisqu'à l’heure du « démarchage » massif pour beaucoup de produits de consommation courants, les potentiels clients sont de plus en plus excédés par ces pratiques.")
            st.write("----------------------------")
            st.image("housing yes.jpg")
            st.image("housing no.jpg")
            st.write("Ce type de graphique, adapté aux variables dichotomiques, représente le même type d’analyse que celui vu précédemment, mais cette fois-ci avec une représentation sous forme de box plot permettant de différencier les deux valeurs de la variable étudiée (ici « housing »). Ainsi les valeurs 1 et 0 en abscisse représentant respectivement la souscription ou non de notre client au produit financier. Les boîtes bleues nous informent quant à elles, sur l’ensemble des valeurs de probabilités prises par le modèle pour chacune des deux valeurs de la variable étudiée. Au vu du graphique, quelles que soient ses valeurs, la variable « housing » n’affectera pas beaucoup le processus de décision de notre modèle. Autrement dit, le fait que le client soit propriétaire ou non, ne change pas beaucoup les chances de souscription à notre produit. Nous vous présentons également pour information, le graphique de l’itération « NO » de cette variable.")
            st.write("----------------------------")
            st.image("contact cellular.jpg")
            st.image("contact telephone.jpg")
            st.image("contact unknown.jpg")
            st.write("En comparant ces trois graphiques, nous pouvons observer que lorsque nous connaissons le mode de contact (classes différentes de « unknown »), il semblerait que les clients ont plus de chance de souscrire au produit. En d’autres termes, appeler le client soit par téléphone fixe, soit par téléphone mobile semble augmenter les chances de souscription.")
            st.write("Nous ne connaissons pas la nature exacte du mode de contact représenté par l’itération « unknown », nous pouvons imaginer que c’est un mode de contact différent que le téléphone. Si c’est le cas, ce mode de contact est à éviter, puisque même si la différence n’est pas très marquée, le client à moins de chance de souscrire lorsque nous ne connaissons pas le mode de contact.")      
            st.write("Par ailleurs, si on se réfère à nouveau au graphique de la partie A), on observe que la plus forte souscription au produit en termes de contact, est réalisée lorsque le client est appelé par téléphone mobile.")
        
        if st.button("Interprétation avec SHAP"):
            st.subheader("Interprétation avec SHAP")
            st.write("**Le graphique de densité**")
            st.image("shap densite.jpg")
            st.write("Le premier graphique que nous vous proposons ci-dessus, représente la densité des valeurs calculées de SHAP pour chaque variable.Il y a essentiellement trois choses à savoir pour interpréter ce type de graphique : plus la ligne de couleur va être longue plus la variable associée va être influente sur l’ensemble des prédictions. En revanche, plus cette ligne est épaisse, plus la variable va être influente sur la prédiction finale.")
            st.write("-	Sans surprise nous remarquons que la variable « duration » est la plus influente sur nos prédictions. Elle a également beaucoup d’impact sur la plupart des prédictions.")
            st.write("-	La variable « housing_no » a un impact sur les prédictions mais pas sur celles qui ont une faible valeur ou une forte valeur.")
            st.write("-	Concernant la variable « poutcome_success », cette variable à un impact sur les prédictions avec une forte valeur mais pas sur celles avec une faible valeur.")
            
            st.write("---------------------------")
            st.write("**Les graphiques de force**")
            st.write("Les graphiques interactifs « stacked force plot » représentent l'influence d'une variable sur l'ensemble des prédictions délivrées par notre modèle. Ces graphiques sont également intéressants puisque, si vous faites circuler le curseur sur le graphique, vous pourrez avoir accès à l'ensemble des valeurs de la variable étudiée.")
            st.write("**la variable 'duration'**")
            st.image("inter duration.png")
            st.write("Avec ce premier graphique nous analysons la variable duration qui est la plus influente de nos variables. On peut remarquer et confirmer au premier coup d'œil l'analyse qui a été faite avant dans ce rapport à savoir que la variable duration doit avoir une valeur haute ou tout du moins minimum afin que les chances de souscription de notre client soient les meilleures. ")
            st.write("**la variable 'day'**")
            st.image("inter day.png")
            st.write("Nous avons vu dans le graphique représentant les variables importantes dans les prédictions de notre modèle que la variable « day » est la seconde variable en termes d'influence sur les prédictions. Nous remarquons et nous confirmons aisément que la variable « Day » est assez similaire en termes d'influence quelles que soient ses valeurs avec on peut toutefois le noter, une légère hausse des probabilités que notre client souscrive à notre modèle sur les valeurs inférieur à moins 1,5.")
            st.write("**la variable 'pdays'**")
            st.image("inter pdays.png")
            st.write("Voici le graphique de la variable « pdays ». Ce graphique est intéressant puisque quelles que soient les valeurs que prend la variable « pdays » on remarque que la probabilité que les clients ne souscrivent pas à notre produit est assez similaire.  En revanche on remarque très nettement, une hausse des probabilités que le client souscrive à notre produit sur les valeurs basses de la variable « pdays ». Autrement dit il semblerait qu'il faille laisser quelques jours entre les deux derniers appels de la banque à notre client, mais si nous en laissons trop nous augmentons les chances que notre client ne souscrive pas à notre produit. Il est important de le noter et possiblement de le réserver dans les recommandations que nous ferons à notre client. Cette variable est intéressante puisqu'avec ces valeurs nous avons une bonne indication sur le côté CRM de la relation client.")
            
            st.write("----------------------------------")
            st.write("**les graphiques de dépendance**")
            st.image("shap dependence balance duration.jpg")
            st.write("Les graphiques de dépendance des variables sont aussi très intéressants en termes d’analyse, puisqu’en plus d’analyser l’influence d’une variable, ils permettent de mettre en relation deux variables entre elles, avec un dégradé de couleur allant du bleu au rouge pour la variable située en ordonnée qui indique le niveau de probabilité de souscription (en rouge) ou de non souscription (en bleu). Ainsi, dans cette représentation, nous mettons en relation les variables « duration » et « balance », et nous pouvons essentiellement remarquer deux choses : l’influence de la variable « duration » grandissante au fur et à mesure que ses valeurs augmentent (en d’autres termes, que la durée de l’appel augmente). Par ailleurs, les valeurs de la variable « balance » sont bien réparties au sein des valeurs de « duration », ce qui se traduit par le fait que le solde du client ne semble pas avoir beaucoup d’influence sur la durée de l’appel.")
            st.image("duration campaign dependance shap.png")
            st.write("Voici la représentation entre les variables « campaign » et « duration ». On remarque facilement un regroupement des plus hautes valeurs de la variable « duration » quand la variable « campaign » a, au contraire, des valeurs faibles. En d’autres termes, la variable « duration » est plus dépendante de la variable « campaign » lorsque cette dernière à des valeurs faibles. Cette analyse vient corroborer celle faite précédemment, à savoir, que la variable « campaign », sur ces valeurs comprises entre 1 et 5, est plus représentatives que sur toutes ses autres valeurs.  Fait intéressant, on remarque également que la probabilité de souscription par le client est bien plus élevée lorsque la valeur de « campaign » est faible, et donc lorsque le client a été contacté peu de fois (0 ou une seule fois) durant la campagne marketing. ")
#############################################################################################################
elif selected == 'Conclusion':
    st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Conclusion</h3>', unsafe_allow_html=True)
    st.write("-----------------------------------------------------------------------------")
    
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Recommandations Client:</h3>', unsafe_allow_html=True)
    st.write("- La classe d'âge.")
    st.write("- La profession.")
    st.write("- La durée de l'appel.")
    st.write("- Contact client.")
    st.write("- Le nombre de contact client.")
    st.write("- Le type de contact client.")
    
    st.write("---------------------------------------------------------------------------------------")
    
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Perspectives Techniques:</h3>', unsafe_allow_html=True)
    st.write("- Features Engineering.")
    st.write("- Amélioration des algorythmes.")

#############################################################################################################
elif selected == 'Démonstration':
    st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Modélisation</h3>', unsafe_allow_html=True)
    st.subheader("Dans ce module, nous allons étudier votre cas avec notre modèle de prédiction.")
    st.write("Afin de commencer votre étude, nous vous invitons à renseigner vos informations ci_après:")
    
    st.write(f'### Vos Informations')
    age = st.selectbox("Quel est votre âge ?", ('-20ans', '20-35ans', '35-50ans', '50-65ans', '65-80ans', '+80ans'))
    education = st.selectbox("Quel est votre niveau d'étude ?", ("tertiary", "secondary", "unknown", "primary"))
    job = st.selectbox("Quel est votre profession ?", ('management', 'technician', 'entrepreneur', 'blue-collar', 'unknown', 'retired', 'admin.', 'services', 'self-employed', 'unemployed', 'housemaid', 'student'))
    marital = st.selectbox("Quel est votre statut marital ?", ('married', 'single', 'divorced'))
    balance = st.slider('Quel est le solde de votre compte en banque ?', -1000, 10000, 1)
    housing = st.selectbox("Etes-vous propriétaire ?", ('yes', 'no'))
    loan = st.selectbox("Avez-vous un crédit en cours ?", ('yes', 'no'))
    default = st.selectbox("Avez-vous déjà eu un défaut de paiement ?", ('yes', 'no'))
    contact = st.selectbox("Par quel moyen avez-vous été contacté par votre banque ?", ('unknown', 'cellular', 'telephone'))
    month = st.selectbox("Quel mois avez-vous été contacté par votre banque, pour la dernière fois ?", ('jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
    day = st.slider("Quel jour avez-vous été contacté par votre banque, pour la dernière fois  ?", 1, 31, 1)
    duration = st.slider("Quelle est la durée, en seconde, de votre dernier contact avec votre banque ?", 0, 600, step=1)
    campaign = st.slider("Combien de fois avez-vous été contacté par votre banque lors de la campagne?", 0,20,1)
    pdays = st.slider("Combien de jours ce sont écoulés depuis le dernier appel de votre banque?", 0,400,1)
    previous = st.slider("Lors de la précédente campagne marketing, combien de fois avez-vous été appélé par votre banque", 0,10,1)
    poutcome = st.selectbox("Avez-vous souscris à l'offre lors de la dernière campagne marketing de votre banque ?", ('other', 'success', 'failure'))
    
    st.write(f'### Récapitulatif')
    st.write("Votre âge est :", age)
    st.write("Votre profession est :", job)
    st.write("Votre niveau d'étude est:", education)
    st.write("Votre statut marital est :", marital)
    st.write("Le solde de votre compte en banque est :"+ str(balance))
    st.write("Vous êtes propriétaire :", housing)
    st.write("Vous avez un crédit en cours :", loan)
    st.write("Avez-vous déjà eu un défaut de paiement :", default)
    st.write("Quel a été le moyen utilsé par votre banque pour vous contacter lors de votre dernier entretien :", contact)
    st.write("Vous avez été contacté par votre banque en :", month)
    st.write("Vous avez été contacté par votre banque le :" +str(day))
    st.write('La durée de votre derner contact avec votre banque est de:'+ str(duration))
    st.write("Le nombre d'appels que vous avez eu lors de la campagne est de :" +str(campaign))
    st.write("Le nombre de jour entre les deux derniers contacts avec votre banque est de :" +str(pdays))
    st.write("Le nombre de contact que vous avez eu lors de la dernière campagne est de :" +str(previous))
    st.write("Avez-vous souscris lors de la dernière campagne :", poutcome)
    
    # Créer un dataframe récapitulatif des données du prospect
    infos_prospect = pd.DataFrame({'age':age, 
                                   'job':job, 
                                   'marital':marital, 
                                   'education':education, 
                                   'default':default,
                                   'balance':balance, 
                                   'housing':housing, 
                                   'loan':loan, 
                                   'contact':contact,
                                   'day':day,
                                   'month':month, 
                                   'duration':duration,
                                   'campaign':campaign,
                                   'pdays':pdays,
                                   'previous':previous,
                                   'poutcome':poutcome}, index=['99999'])
    st.write("  ")
    st.write(f'### Voici le tableau avec vos informations')
    st.dataframe(infos_prospect.head())
    
    if st.button("Auriez-vous souscris à notre produit ?", key="classify"):
        st.subheader("Résultats de l'éxecution :")
        
        data=pd.read_csv('dataset-avant-encodage')
        data=data.drop('Unnamed: 0', axis=1)
        data_final = pd.concat([data, infos_prospect], axis=0)
        explicatives = data_final.drop('deposit', axis=1)
        explicatives_num = explicatives.select_dtypes(include=['float64', 'int64'])
        explicatives_cat = explicatives.select_dtypes(include=['object', 'category'])
        #Encodage
        enc_num = StandardScaler().fit(explicatives_num)
        encoded_explicatives_num = enc_num.transform(explicatives_num)
        explicatives_num_encoded = pd.DataFrame(encoded_explicatives_num, columns=enc_num.get_feature_names_out())
        enc = OneHotEncoder(sparse=False).fit(explicatives_cat)
        encoded = enc.transform(explicatives_cat)
        explicatives_cat_encoded = pd.DataFrame(encoded, columns=enc.get_feature_names_out())
        explicatives_encoded = pd.concat([explicatives_cat_encoded, explicatives_num_encoded], axis=1)
        data_prospect = explicatives_encoded.iloc[-1:].T.T
        st.write(data_prospect)
        
    
        model = load('model_final.joblib')

        #Affichage de la probabilité de souscription
        proba = model.predict_proba(data_prospect)
        proba_souscription = proba[0][1]
        proba_non_souscription = proba[0][0]
        st.write('--------------------------------------------------')
        st.markdown('<h3 style="color:black;font-weight:bold;font-size:25px;">Voici vos résultats:</h3>', unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; color:green;'>Probabilité de souscription : {:.2f}%</p>".format(proba_souscription * 100), unsafe_allow_html=True)
        st.markdown("<p style='font-size:15px; color:green;'>Probabilité de non souscription : {:.2f}%</p>".format(proba_non_souscription * 100), unsafe_allow_html=True)

###########################################################################################################
elif selected == 'Faites votre Modélisation':
    st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:35px;">Modélisation</h3>', unsafe_allow_html=True)
    st.subheader("Dans ce module vous allez pouvoir faire votre modèlisation")
    st.write(":red[Information de Navigation.]")
    st.write("Afin de naviguer entre les sous-menus, nous vous invitons à selectionner la rubrique que vous souhaitez dans le menu de gauche: Menu Data Visualisation.")
    st.write("Nous vous informons que dans un souci de rapidité et de fluidité d'affichage des résultats, toutes les modèlisations seront réslisées sur le jeu de données original encodé et non rééchantillonné.")
        
        # Définition des FONCTIONS
        
        # Intanciation du dataset
    def chargement_dataset_SVM():
        df = pd.read_csv('Dataset_complet_encoded')
        df.drop('Unnamed: 0', axis=1)
        df = df.head(20000)
        explicatives = df.drop('deposit', axis=1)
        target = df['deposit']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(explicatives, target, test_size=0.35, random_state=42)
        return X_train, X_test, y_train, y_test
        
    def chargement_dataset_autres():
        df = pd.read_csv('Dataset_complet_encoded')
        df.drop('Unnamed: 0', axis=1)
        df = df.head(30000)
        explicatives = df.drop('deposit', axis=1)
        target = df['deposit']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(explicatives, target, test_size=0.35, random_state=42)
        return X_train, X_test, y_train, y_test
        
        # Graphiques 
    def graphiques(graphes):
        if "Matrice de Confusion" in graphes:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(20,12))
            cm_forest = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm_forest,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
            plt.title('Matrice de Confusion', fontsize=35)
            plt.ylabel('Valeur Réelle',fontsize=30)
            plt.xlabel('Valeur Prédites',fontsize=30)
            st.pyplot(fig)
            st.write("--------------------------------------------------")
                
        if "CURVE ROC" in graphes:
            from sklearn.metrics import roc_curve, auc
            fig, ax = plt.subplots(figsize=(20,12))
            # fpr taux de vrais positifs tpr taux de faux positifs
            fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)
            # Affichage de la courbe de performance du modèle
            plt.plot(fpr, tpr, color='orange', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('Coube ROC',fontsize=35)
            plt.xlabel('Taux faux positifs',fontsize=30)
            plt.ylabel('Taux vrais positifs',fontsize=30)
            st.pyplot(fig)
            st.write("------------------------------------")
                
                
    def graphiques_tree(graphes):
         if "Matrice de Confusion" in graphes:
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            fig, ax = plt.subplots(figsize=(20,12))
            cm_forest = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm_forest,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
            plt.title('Matrice de Confusion',fontsize=35)
            plt.ylabel('Valeur Réelle',fontsize=30)
            plt.xlabel('Valeur Prédites',fontsize=30)
            st.pyplot(fig)
            st.write("--------------------------------------")
                
         if "CURVE ROC" in graphes:
            from sklearn.metrics import roc_curve, auc
            fig, ax = plt.subplots(figsize=(20,12))
            # fpr taux de vrais positifs tpr taux de faux positifs
            fpr, tpr, seuils = roc_curve(y_test, probs[:,1], pos_label=1)
            roc_auc = auc(fpr, tpr)

            # Affichage de la courbe de performance du modèle
            plt.plot(fpr, tpr, color='orange', lw=2, label='Modèle clf (auc = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aléatoire (auc = 0.5)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.title('Coube ROC',fontsize=35)
            plt.xlabel('Taux faux positifs',fontsize=30)
            plt.ylabel('Taux vrais positifs',fontsize=30)
            st.pyplot(fig)
            st.write("--------------------------------------")
                
         if "DecisionTree" in graphes:
            from sklearn.tree import plot_tree
            fig, ax = plt.subplots(figsize=(20, 20))  
            plot_tree(model,
                      class_names = ['Yes','No'],
                      filled = True, 
                      rounded = True)
            st.pyplot(fig)
            st.write("--------------------------------------")
        
        
        #Sélection du SVM
    selection_modele = st.sidebar.selectbox('Sélectionner votre modèle', ("SVM", "Logistic Regression", "KNeighbors", "RandomForest", "DecisionTree"))
    if selection_modele=='SVM':
            st.subheader('Vous avez choisi le modèle SVM')
            st.write("Il est important de vous signaler qu'en raison de la durée d'éxecution du modèle SVM nous allons raccourcir le jeu de données sur lequel nous allons entrainer et prédire les données.")
            st.write("Le jeu de donnée original est de 45000 lignes celui utiliser pour cette modèlisation sera de 20000 lignes.")
            
            #Selection des Hyper-Paramètres
            kernel_selection = st.sidebar.selectbox("Sélectionner le kernel de votre choix", ('rbf', 'linear'))
            
            #Selection du parametre de regulation
            c_selection = st.sidebar.number_input("Sélectionner le paramètre de Régulation", 0.1, 20.0, step=0.2)
            
            #Selection du gamme
            gamma_selection = st.sidebar.number_input("Sélectionner la valeur de l'argument Gamma", 0.001, 1.0, step=0.005)
            
            #Sélection des Graphiques
            graphiques_performances = st.sidebar.multiselect("Quels graphiques souhaitez-vous pour visualiser la performance", ("CURVE ROC", "Matrice de Confusion"))
            
            #Execution
            if st.sidebar.button("Exécution", key="classify"):
                st.subheader("Résultats de l'éxecution :")
                
                X_train, X_test, y_train, y_test = chargement_dataset_SVM()
                
                from sklearn.svm import SVC
                model = SVC(kernel = kernel_selection,
                            C = c_selection,
                            gamma = gamma_selection,
                            probability=True,
                            class_weight = 'balanced')
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                # Metriques
                from sklearn.metrics import recall_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import precision_score
                
                st.write("Accuarcy", (accuracy_score(y_test, y_pred).round(2)*100), "%")
                st.write("Precision", precision_score(y_test, y_pred).round(2)*100, "%")
                st.write('F1', f1_score(y_test, y_pred).round(2)*100, "%")
                st.write('Rappel', recall_score(y_test, y_pred).round(2)*100, "%")
                
                #Affichage Graphiques
                graphiques(graphiques_performances)
        
    if selection_modele=='Logistic Regression':
            st.subheader('Vous avez choisi le modèle Régréssion Logistique')
            st.write("Il est important de vous signaler qu'en raison de la durée d'éxecution du modèle SVM nous allons raccourcir le jeu de données sur lequel nous allons entrainer et prédire les données.")
            st.write("Le jeu de donnée original est de 45000 lignes celui utiliser pour cette modèlisation sera de 30000 lignes.")
            
            #Selection des Hyper-Paramètres
            solver_selection = st.sidebar.selectbox("Sélectionner le solver de votre choix", ('lbfgs', 'liblinear'))
            
            #Selection du parametre de regulation
            max_iter_selection = st.sidebar.number_input("Sélectionner le nombre maximum d'itérations :", 50.0, 2000.0, step=100.0)

            #Sélection des Graphiques
            graphiques_performances = st.sidebar.multiselect("Quels graphiques souhaitez-vous pour visualiser la performance", ("CURVE ROC", "Matrice de Confusion"))
            
            #Execution
            if st.sidebar.button("Exécution", key="classify"):
                st.subheader("Résultats de l'éxecution :")
                
                X_train, X_test, y_train, y_test = chargement_dataset_autres()
                
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(solver = solver_selection,
                                           max_iter = max_iter_selection,
                                           class_weight = 'balanced')
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                # Metriques
                from sklearn.metrics import recall_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import precision_score
                
                st.write("Accuarcy", (accuracy_score(y_test, y_pred).round(2)*100), "%")
                st.write("Precision", precision_score(y_test, y_pred).round(2)*100, "%")
                st.write('F1', f1_score(y_test, y_pred).round(2)*100, "%")
                st.write('Rappel', recall_score(y_test, y_pred).round(2)*100, "%")
                
                #Affichage Graphiques
                graphiques(graphiques_performances)
                
        
    if selection_modele=='KNeighbors':
            st.subheader('Vous avez choisi le modèle KNeighbors')
            st.write("Il est important de vous signaler qu'en raison de la durée d'éxecution du modèle SVM nous allons raccourcir le jeu de données sur lequel nous allons entrainer et prédire les données.")
            st.write("Le jeu de donnée original est de 45000 lignes celui utiliser pour cette modèlisation sera de 30000 lignes.")
            
            #Selection des Hyper-Paramètres
            n_neighbors_selection = st.sidebar.number_input("Sélectionner le nombre de voisins de votre choix", 5, 25, step=5)
            
            #Selection du parametre de regulation
            metric_selection = st.sidebar.selectbox("Sélectionner la métrique :", ('minkowski', 'manhattan', 'chebyshev'))

            #Sélection des Graphiques
            graphiques_performances = st.sidebar.multiselect("Quels graphiques souhaitez-vous pour visualiser la performance", ("CURVE ROC", "Matrice de Confusion"))
            
            #Execution
            if st.sidebar.button("Exécution", key="classify"):
                st.subheader("Résultats de l'éxecution :")
                
                X_train, X_test, y_train, y_test = chargement_dataset_autres()
                
                from sklearn.neighbors import KNeighborsClassifier
                model = KNeighborsClassifier(n_neighbors = n_neighbors_selection,
                                             metric = metric_selection,
                                             weights = 'distance')
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                # Metriques
                from sklearn.metrics import recall_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import precision_score
                
                st.write("Accuarcy", (accuracy_score(y_test, y_pred).round(2)*100), "%")
                st.write("Precision", precision_score(y_test, y_pred).round(2)*100, "%")
                st.write('F1', f1_score(y_test, y_pred).round(2)*100, "%")
                st.write('Rappel', recall_score(y_test, y_pred).round(2)*100, "%")
                
                #Affichage Graphiques
                graphiques(graphiques_performances)
                
                
    if selection_modele=='RandomForest':
            st.subheader('Vous avez choisi le modèle RandomForest')
            st.write("Il est important de vous signaler qu'en raison de la durée d'éxecution du modèle SVM nous allons raccourcir le jeu de données sur lequel nous allons entrainer et prédire les données.")
            st.write("Le jeu de donnée original est de 45000 lignes celui utiliser pour cette modèlisation sera de 30000 lignes.")
            
            #Selection des Hyper-Paramètres
            n_estimators_selection = st.sidebar.number_input("Sélectionner le nombre d'arbre' :", 50, 2000, step=100)
            
            criterion_selection = st.sidebar.selectbox("Sélectionner le criterion :", ('gini', 'entropy'))
            
            max_depth_selection = st.sidebar.number_input("Sélectionner la profondeur de l'arbre", 1, 25, step=1)

            #Sélection des Graphiques
            graphiques_performances = st.sidebar.multiselect("Quels graphiques souhaitez-vous pour visualiser la performance", ("CURVE ROC", "Matrice de Confusion"))
            
            #Execution
            if st.sidebar.button("Exécution", key="classify"):
                st.subheader("Résultats de l'éxecution :")
                
                X_train, X_test, y_train, y_test = chargement_dataset_autres()
                
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators = n_estimators_selection,
                                               criterion = criterion_selection,
                                               max_depth = max_depth_selection)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                # Metriques
                from sklearn.metrics import recall_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import precision_score
                
                st.write("Accuarcy", (accuracy_score(y_test, y_pred).round(2)*100), "%")
                st.write("Precision", precision_score(y_test, y_pred).round(2)*100, "%")
                st.write('F1', f1_score(y_test, y_pred).round(2)*100, "%")
                st.write('Rappel', recall_score(y_test, y_pred).round(2)*100, "%")
                
                #Affichage Graphiques
                graphiques(graphiques_performances)
                
                
    if selection_modele=='DecisionTree':
            st.subheader('Vous avez choisi le modèle DecisionTree')
            st.write("Il est important de vous signaler qu'en raison de la durée d'éxecution du modèle SVM nous allons raccourcir le jeu de données sur lequel nous allons entrainer et prédire les données.")
            st.write("Le jeu de donnée original est de 45000 lignes celui utiliser pour cette modèlisation sera de 30000 lignes.")
            
            #Selection des Hyper-Paramètres           
            criterion_selection = st.sidebar.selectbox("Sélectionner le criterion :", ('gini', 'entropy'))
            
            max_depth_selection = st.sidebar.number_input("Sélectionner la profondeur de l'arbre", 1, 25, step=1)
            
            #min_samples_split_selection = st.sidebar.slider("Sélectionner le nombre maximum de feuille", 1.0, 25.0, 1.0)
            
            max_features_selection = st.sidebar.selectbox("Sélectionner les features :", ('log2', 'sqrt'))

            #Sélection des Graphiques
            graphiques_performances = st.sidebar.multiselect("Quels graphiques souhaitez-vous pour visualiser la performance", ("CURVE ROC", "Matrice de Confusion", "DecisionTree"))
            
            #Execution
            if st.sidebar.button("Exécution", key="classify"):
                st.subheader("Résultats de l'éxecution :")
                
                X_train, X_test, y_train, y_test = chargement_dataset_autres()
                
                from sklearn.tree import DecisionTreeClassifier
                model = DecisionTreeClassifier(criterion = criterion_selection,
                                               max_depth = max_depth_selection,
                                               #min_samples_split = min_samples_split_selection,
                                               max_features = max_features_selection)
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                probs = model.predict_proba(X_test)
                
                # Metriques
                from sklearn.metrics import recall_score
                from sklearn.metrics import accuracy_score
                from sklearn.metrics import f1_score
                from sklearn.metrics import precision_score
                
                st.write("Accuarcy", (accuracy_score(y_test, y_pred).round(2)*100), "%")
                st.write("Precision", precision_score(y_test, y_pred).round(2)*100, "%")
                st.write('F1', f1_score(y_test, y_pred).round(2)*100, "%")
                st.write('Rappel', recall_score(y_test, y_pred).round(2)*100, "%")
                
                #Affichage Graphiques
                graphiques_tree(graphiques_performances)
##################################################################################################################################
elif selected == 'Pour aller plus loin':
    st.markdown('<h3 style="color:mediumblue;font-weight:bold;font-size:30px;">Pour aller plus loin</h3>', unsafe_allow_html=True)
    st.subheader("Regard critique")
    st.write("Notre projet est maintenant terminé, nous avons répondu à la problématique de notre client.")
    st.write("En revanche nous nous devons de prendre du recul et d'exposer les axes d'améliorations concernant ce projet. Après reflexion, nous vous en proposons plusieurs:")
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Le Jeu de Données:</h3>', unsafe_allow_html=True)
    st.write("La première possibilité d'amélioration concerne le jeu de données. En effet celui-ci étant très déséquilibré, il serait interessant d'avoir plus d'informations sur les clients, (plus de variables par client), afin de mieux les détecter. Il serait également intéressant d'augmenter le nombres d'observations afin d'avoir plus de clients dans le jeu de données.")
    st.write("---------------------------------------------------------------")
    st.markdown('<h3 style="color:green;font-weight:bold;font-size:20px;">Les Données Complémentaires:</h3>', unsafe_allow_html=True)
    st.write("Un autre paramètre à prendre en compte pour aller plus loin dans ce projet serait, **les données complémentaires**. Il aurait été intéressant de savoir quels étaient **les taux d'intérêts** pratiqués par la banque lors de la campagne. Nous savons que les taux d'intérêts des banques ont un impact sur la consommation des français. De même il aurait été intéressant de connaître **le taux de rentabilité** du dépôt à terme. Enfin, d'un point de vue plus général, il aurait été important de connaître **l'année durant laquelle la banque a pratiqué sa campagne et avoir également, plusieurs années de données pour les comparer**. Nous connaissons tous l'année 2020 qui a été catastrophique sur le plan économique pour la plupart des société et qui a augmenté la méfiance des français dans l'investissement en général.")
    st.write("---------------------------------------------------------------")
    
###############################################################################################################################"
elif selected == 'Remerciements':
    st.markdown('<h3 style="color:darkblue;font-weight:bold;font-size:35px;">Remerciements</h3>', unsafe_allow_html=True)
    st.write("Nous tenons à vous remercier pour l'intéret porté à notre projet de fin de formation. Nous remercions également Yohan pour ses conseils précieux.")
    st.write('------------------')
    st.write("Si vous souhaitez nous rencontrer, n'hésitez pas à nous contacter.")
    st.write('Auteurs:')
    st.write("[Benjamin Chartier](https://www.linkedin.com/in/benjamin-chartier-data-analyst-python-machinelearning-powerbi)")
    st.write("[Sarah Diouri]( https://www.linkedin.com/in/sarah-diouri-7396247b/)")
    st.write("[Thibault](https://fr.linkedin.com/in/thibault-v-varin-7245b171)")
    st.write("[Balde](https://www.linkedin.com/in/saliou-balde-data-analyste-python/)")
    st.write('------------------')
    st.image("datascientest_5fe203c15886e.jpg")
    st.write("Formation Data Analyst - Janvier 2023") 
    st.write('------------------')
    st.image('image_bank_streamlit.jpg')
    