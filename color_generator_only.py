####################################################################################################
################################ LIBRARIES, VARIABLES, AND FUNCTIONS ###############################
####################################################################################################
############################################
######### DATA COLLECTION LIBRARIES ########
# Scraping Libraries
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
import time
# Regex Library
import re
import time
############################################
######### DATA PROCESSING LIBRARIES ########
from colormath.color_objects import LabColor, LCHabColor, XYZColor, sRGBColor
from colormath.color_conversions import convert_color
from PIL import Image
import colormap
import pandas as pd
# For visualizations
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
# import tkinter
# tkinter._test()
# import nest_asyncio
# nest_asyncio.apply()
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sin, cos, radians
import seaborn as sns
from G_to_J import list_clustering_graph_to_json, list_clustering_graph_preproccessing
import webbrowser, os
from pathlib import Path
# For clustering
from colormath.color_diff import delta_e_cmc
import chart_studio.plotly as py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from collections import Counter
import collections
import networkx as nx
# For Graph Traversal
import random
from colormath.color_diff import delta_e_cie2000
# For colormind integration
import ast
import requests
import seaborn as sns

############################################
######## KOHONEN SOM-RELATED IMPORTS #######
# import KohonenMap
# import GeneratePoints

############################################
############# MASTER VARIABLES #############
# Data Processing Hyperparameters
FULL_SCROLL_NUMBER = 10
ADOBE_FULL_SCROLL_NUMBER = 3400
COLOR_LOCATION = "explore-palette_colors"
COLOR_LOCATION_ADOBE = "Theme__theme___2NcED"
# CUBIC_THRESHOLD = 5
GLOBAL_FIG_SIZE = 40
# K_MAX = 800
K = 3000
TOP_CONST = [0, 10, 25, 50]
NUM_COLORS = 8
NUM_PALETTES = 15
distinct_colors = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58", "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176", "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C", "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433", "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E", "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]
distinct_colors = distinct_colors * (int(K/len(distinct_colors)) + 1)

# COMBINATIONS (white not exclued yet) #
# DIVERSITY = False, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'HISTORICAL MAX' --> Lots of white, some very purple
# DIVERSITY = False, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'MAX'            --> Lots of white, initial 1/2/3 non-white colors
# DIVERSITY = False, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'HISTORICAL MAX'         --> A little more muted, some low-L colors
# DIVERSITY = False, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'MAX'                    --> A little more muted, some low-L colors
# DIVERSITY = True, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'HISTORICAL MAX'  --> Lots of oscillation (turquoise and red)
# DIVERSITY = True, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'MAX'             -->
# DIVERSITY = True, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'HISTORICAL MAX'
# DIVERSITY = True, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'MAX'
DIVERSITY = True
if(DIVERSITY):
    MAX_CENTROID_ACCESS = 2
else:
    MAX_CENTROID_ACCESS = None
COLOR_PICK_METHOD = 'REPRESENTATIVE'
CENTROID_PICK_METHOD = 'MAX'
WHITE_LAB = LabColor(100, 0, 0)
WHITE_THRESH = 10

# AffinityNet Master Variables
buttons = False                 # clustering options for the graph
groups_attribute = True         # specify group for each node for auto-coloring
groups_option = True            # Default coloring already takes place, extra clustering functions are absent/inactive
physics_option = True           # specify physics for graph motion
node_labels = True              # specify labels for each node
manipulation_option = True      # specify manipulation values
interaction_option = True       # specify interaction values
layout_option = True            # specify layout appearance
shape_name = 'dot'              # shape of each node
shapes = True                   # specify if shape should be set
node_values = False             # specify node size (won't do anything though)
heading_format = True           # Heading of file showing details of Graph
VIS_parameters = [buttons, groups_attribute, groups_option, physics_option, node_labels, manipulation_option, interaction_option, layout_option, shapes, node_values, heading_format]
VIS_location = '/Users/Ray/Documents/Python/Glencoe/'
OPEN_TAB = False                # Open page in chrome?

############################################
############# PROCESS FUNCTIONS ############
def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    # print('Track - data_for_cylinder_along_z')
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2 * np.pi, 50)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = radius * np.cos(theta_grid) + center_x
    y_grid = radius * np.sin(theta_grid) + center_y
    return x_grid, y_grid, z_grid
# Plots a 3d scatter plot
def plot3D_object(figure_size, title, labels, limits, df, color_map, cyl, keys):
    print('TRACK - plot3D_object')
    if(keys is None):
        keys = labels
    fig = plt.figure(figsize = (figure_size, figure_size))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_title(title)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])
    ax.set_zlim(limits[2][0], limits[2][1])
    if cyl:
        Xc, Yc, Zc = data_for_cylinder_along_z((limits[0][0] + limits[0][1])/2.0,(limits[1][0] + limits[1][1])/2.0, min(limits[0][0], limits[0][1], limits[1][0], limits[1][1]), limits[2][1])
        ax.plot_surface(Xc, Yc, Zc, alpha = 0.2)
    ax.scatter(df[keys[0]], df[keys[1]], df[keys[2]], c = color_map, s = 60)

    return ax
# Makes HTML for AffinityNet visualization
def populate_files_ML(browser_open, twod_list, K, top_CONST, VIS_parameters, shape_name, VIS_location, init_strat_name = '', prim_dir_name = '', desc = ''):
    print('TRACK - populate_files_ML')
    HTML_FILES = []
    HTML_FILE_NAMES = []
    PRIMARY_DIRECTORY_NAMES = []

    if(desc == ''):
        desc = prim_dir_name.replace('/', '')

    dict_stratClass_edgeClass, edge_weights_3t, d_keys_ind, d_values_ind, color_rules = list_clustering_graph_preproccessing(twod_list, K, top_CONST)

    HTML_OUTPUT = list_clustering_graph_to_json(desc, K, top_CONST, VIS_parameters, shape_name, dict_stratClass_edgeClass, d_keys_ind, d_values_ind, edge_weights_3t, color_rules, VIS_location, init_strat_name)
    HTML_NAME = "Coolors_AUTO-" + 'ALL' + "##" + 'ALL' + "-" + str(top_CONST) + ".html"
    DIRECTORY_NAME = str(top_CONST)

    HTML_FILES.append(HTML_OUTPUT)
    HTML_FILE_NAMES.append(HTML_NAME)
    PRIMARY_DIRECTORY_NAMES.append(DIRECTORY_NAME)

    # Create directories, write to file, open in default broswer [chrome]
    for iter_num in range(len(HTML_FILES)):
        most_specific_path = str(prim_dir_name) + str(PRIMARY_DIRECTORY_NAMES[iter_num]) + "/" + str(HTML_FILE_NAMES[iter_num])
        if not os.path.exists(str(prim_dir_name + PRIMARY_DIRECTORY_NAMES[iter_num])):
            os.makedirs(str(prim_dir_name + PRIMARY_DIRECTORY_NAMES[iter_num]))
        if not Path(most_specific_path).is_file():
            with open(most_specific_path, "w") as file:
                file.write(HTML_FILES[iter_num])
        if(browser_open):
            webbrowser.open_new_tab("file://" + os.path.realpath(most_specific_path))

    return 0
# Returns k-location of specific color
def color_location(df_annotated_RGB, color_3_tuple):
    # print('Track - color_location')
    return list(chain.from_iterable(df_annotated_RGB.loc[(round(df_annotated_RGB['R']) == round(color_3_tuple[0])) & (round(df_annotated_RGB['G']) == round(color_3_tuple[1])) & (round(df_annotated_RGB['B']) == round(color_3_tuple[2]))].drop_duplicates(keep = 'first').to_numpy().tolist()))[3]
# Plot 3D PCA Plot (w/ KMeans clustering)
def plot_PCA(figure_size, title, PCA_points, cluster_labels, kmeans_config):
    # print('TRACK - plot_PCA')
    fig = plt.figure(figsize = (figure_size, figure_size))
    ax_PCA = fig.add_subplot(111, projection = '3d')
    # Plotting all the points in PCA space
    ax_PCA.set_title(title)
    ax_PCA.scatter(PCA_points[:,0], PCA_points[:,1], PCA_points[:,2],
                c = cluster_labels, cmap = 'viridis',
                edgecolor = 'k', s = 40, alpha = 0.5)
    ax_PCA.set_xlabel('First Principal Component')
    ax_PCA.set_ylabel('Second Principal Component')
    ax_PCA.set_zlabel('Third Principal Component')
    # Plotting all the cluster centroids in PCA space
    ax_PCA.scatter(kmeans_config.cluster_centers_[:,0], kmeans_config.cluster_centers_[:,1],
               kmeans_config.cluster_centers_[:,2],
               s = 300, c = 'r', marker='*', label = 'Centroid')
    return ax_PCA
# Execute full PCA process
def complete_PCA_process(PCA_title, df_ORIGINAL, n_comp = 3):
    print('TRACK - complete_PCA_process')
    pca_ = PCA(n_components = n_comp)
    kmeans_PCA = KMeans(n_clusters = K, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 3)
    X_Fit_Gamut_to_PCA = pca_.fit_transform(df_ORIGINAL)
    y_kmeans_PCA = kmeans_PCA.fit_predict(X_Fit_Gamut_to_PCA)
    ax_PCA_Gamut = plot_PCA(GLOBAL_FIG_SIZE, PCA_title, X_Fit_Gamut_to_PCA, y_kmeans_PCA, kmeans_PCA)

    return pca_, kmeans_PCA, X_Fit_Gamut_to_PCA, y_kmeans_PCA, ax_PCA_Gamut
# Display inverse transfored colors with centroid connections
# display_inv_transform_sRGB('LAB-PCA-sRGB', pca_LAB, kmeans_PCA_LAB, X_Fit_CIELAB_to_PCA, y_kmeans_PCA_LAB, RGB_values, plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in sRGB Gamut (w/ LAB-PCA Centroid Connections)', ('R', 'G', 'B'), ((0, 255), (0, 255), (0, 255)), df_RGB, flat_HEX, False, None))
def display_inv_transform_sRGB(pipeline_str, pca_, kmeans_PCA, X_Fit_Gamut_to_PCA, y_kmeans_PCA, RGB_values, sRGB_plot_ORIGINAL):
    print('TRACK - display_inv_transform_sRGB')
    pre_PCA_dimension = pipeline_str[:pipeline_str.index('-')]
    pre_PCA_dimension_list = [dim for dim in pre_PCA_dimension]

    # Convert from specific Gamut to sRGB Space for plotting purposes
    if(pre_PCA_dimension == 'sRGB'):
        df_Gamut_pca_sRGB = pd.DataFrame(pca_.inverse_transform(X_Fit_Gamut_to_PCA.tolist()).tolist())
        df_Gamut_pca_sRGB_CENT = pd.DataFrame(pca_.inverse_transform(kmeans_PCA.cluster_centers_).tolist())
    else:
        Gamut_specific_colors = pca_.inverse_transform(X_Fit_Gamut_to_PCA.tolist()).tolist()
        Gamut_specific_CENT_colors = pca_.inverse_transform(kmeans_PCA.cluster_centers_).tolist()
        sRGB_values_local = []
        sRGB_CENT_values_local = []
        # Convert from LAB/LCH -> sRGB (CIE LAB/LCH -> sRGB). Shouldn't be any overlap between CIE LAB/LCH to sRGB (never traversed beyond sRGB Gamut).
        if(pre_PCA_dimension == 'LAB'):
            sRGB_values_local = [convert_color(LabColor(COLOR[0], COLOR[1], COLOR[2]), sRGBColor, target_illuminant = 'd50').get_value_tuple() for COLOR in Gamut_specific_colors]
            sRGB_CENT_values_local = [convert_color(LabColor(COLOR[0], COLOR[1], COLOR[2]), sRGBColor, target_illuminant = 'd50').get_value_tuple() for COLOR in Gamut_specific_CENT_colors]
        elif(pre_PCA_dimension == 'LCH'):
            sRGB_values_local = [convert_color(LCHabColor(COLOR[0], COLOR[1], COLOR[2]), sRGBColor, target_illuminant = 'd50').get_value_tuple() for COLOR in Gamut_specific_colors]
            sRGB_CENT_values_local = [convert_color(LCHabColor(COLOR[0], COLOR[1], COLOR[2]), sRGBColor, target_illuminant = 'd50').get_value_tuple() for COLOR in Gamut_specific_CENT_colors]
        df_Gamut_pca_sRGB = pd.DataFrame(sRGB_values_local) * 255.0
        df_Gamut_pca_sRGB_CENT = pd.DataFrame(sRGB_CENT_values_local) * 255.0

    df_Gamut_pca_sRGB.columns = ['R', 'G', 'B']
    df_Gamut_pca_sRGB.insert(3, 'Category', y_kmeans_PCA.tolist())
    df_Gamut_pca_sRGB_CENT.columns = ['R', 'G', 'B']

    # Inverse Transform Centroid Points to sRGB Gamut
    # Configure figure object
    fig_centr = plt.figure(figsize = (GLOBAL_FIG_SIZE, GLOBAL_FIG_SIZE))
    ax_Gamut_inverse_transform_CENTROID_PLOTS = fig_centr.add_subplot(111, projection = '3d')
    ax_Gamut_inverse_transform_CENTROID_PLOTS.set_title(pipeline_str + ' Inverse Transformed Points in sRGB Space w/ Clustering and Centroid Palette Connections')
    ax_Gamut_inverse_transform_CENTROID_PLOTS.set_xlabel('R')
    ax_Gamut_inverse_transform_CENTROID_PLOTS.set_ylabel('G')
    ax_Gamut_inverse_transform_CENTROID_PLOTS.set_zlabel('B')
    # Connect the points in each palette in sRGB Space (based on respective centroids)
    sRGB_plot_ORIGINAL.scatter(df_Gamut_pca_sRGB_CENT['R'], df_Gamut_pca_sRGB_CENT['G'], df_Gamut_pca_sRGB_CENT['B'], s = 300, c = 'r', marker = '*', label = 'Centroid')
    c = 0 # For color formatting, and *connection limit
    for palette_colors in RGB_values:
        loc = [[color_location(df_Gamut_pca_sRGB, tuple([[i * 255.0 for i in color][0], [i * 255.0 for i in color][1], [i * 255.0 for i in color][2]]))] for color in palette_colors]
        loc = list(chain.from_iterable(loc))

        CENT_points = [list(chain.from_iterable(df_Gamut_pca_sRGB_CENT.iloc[[loc_i]].to_numpy().tolist())) for loc_i in loc]
        CENT_points_x = [val[0] for val in CENT_points]
        CENT_points_y = [val[1] for val in CENT_points]
        CENT_points_z = [val[2] for val in CENT_points]

        sRGB_plot_ORIGINAL.plot(CENT_points_x, CENT_points_y, CENT_points_z, color = distinct_colors[c % len(distinct_colors)])
        ax_Gamut_inverse_transform_CENTROID_PLOTS.plot(CENT_points_x, CENT_points_y, CENT_points_z, color = distinct_colors[c % len(distinct_colors)])

        c += 1

    # Plot all points + centroids
    ax_Gamut_inverse_transform_CENTROID_PLOTS.scatter(df_Gamut_pca_sRGB['R'], df_Gamut_pca_sRGB['G'], df_Gamut_pca_sRGB['B'],
                c = df_Gamut_pca_sRGB['Category'], cmap = 'viridis',
                edgecolor = 'k', s = 60, alpha = 0.5)
    ax_Gamut_inverse_transform_CENTROID_PLOTS.scatter(df_Gamut_pca_sRGB_CENT['R'], df_Gamut_pca_sRGB_CENT['G'], df_Gamut_pca_sRGB_CENT['B'],
                s = 300, c = 'r', marker = '*', label = 'Centroid')

    return ax_Gamut_inverse_transform_CENTROID_PLOTS, df_Gamut_pca_sRGB, sRGB_plot_ORIGINAL

###########################################
############ UTILITY FUNCTIONS ############
# Check if element is on page
def check_exists_by_xpath(xpath, driver):
    try:
        driver.find_element_by_xpath(xpath)
    except NoSuchElementException:
        return False
    return True
# Converts given HEX code to RGB 3-tuple in sRGB Gamut
def HEX_TO_RGB(hex_input, clamped = False):
    # print('Track - HEX_TO_RGB')
    if(clamped == True):
        return tuple(int(hex_input[i : i + 2], 16)/255.0 for i in (0, 2, 4))
    return tuple(int(hex_input[i : i + 2], 16) for i in (0, 2, 4))
# Retrive edge weights
def graph_from_weights(annotated_df):
    test_graphs = [nx.complete_graph(palette) for palette in annotated_df]
    complete_EDGE_LIST = [tuple(sorted(tup)) for tup in list(chain.from_iterable([w.edges for w in test_graphs]))]
    edge_weights = collections.Counter(complete_EDGE_LIST)
    first_edge_value, second_edge_value = [i[0] for i in list(edge_weights)], [i[1] for i in list(edge_weights)]
    edge_weights_3t = list(zip(first_edge_value, second_edge_value, list(edge_weights.values()))) # based on edge weights (SORT, THEN TRUNCATE)

    G = nx.Graph()
    G.add_weighted_edges_from(edge_weights_3t)

    return G
# Returns multi-dimensional mesh to visualize cylinder in plot
# Equality testing for different color gamuts. Takes rounding of dimesnsions into account.
def compare_RGB(obj1, obj2):
    return round(obj1.rgb_r) == round(obj2.rgb_r) and round(obj1.rgb_g) == round(obj2.rgb_g) and round(obj1.rgb_b) == round(obj2.rgb_g)
def compare_LAB(obj1, obj2):
    return round(obj1.lab_l) == round(obj2.lab_l) and round(obj1.lab_a) == round(obj2.lab_a) and round(obj1.lab_b) == round(obj2.lab_b)
def compare_LCH(obj1, obj2):
    return round(obj1.lch_l) == round(obj2.lch_l) and round(obj1.lch_c) == round(obj2.lch_c) and round(obj1.lch_h) == round(obj2.lch_h)
# Remove occurences of one list in another list
def remove_occurences_LAB(base, comparer):
    new_list = []
    if(len(comparer) == 0):
        return base
    repeats = []
    counter = 0
    for item in base:
        for element in comparer:
            if(compare_LAB(item, element) == True):
                repeats.append(counter)
        counter += 1
    for i in sorted(list(set(repeats)), reverse = True):
        del base[i]
    return base
# Display palette from given RGB List
def display_palette_from_RGB(RGB_colors, clamped = True):
    if(clamped):
        HEX_list = [colormap.rgb2hex(round(r * 255.0), round(g * 255.0), round(b * 255.0)) for r, g, b, in RGB_colors]
    else:
        HEX_list = [colormap.rgb2hex(round(r), round(g), round(b)) for r, g, b, in RGB_colors]
    sns.set_palette(HEX_list)
    sns.palplot(sns.color_palette())
# Remove whites from different gamuts
def remove_whites(lop, init_type):
    final_lop = []
    final_pal = []
    for pal_i in range(len(lop)):
        for color_i in range(len(lop[pal_i])):
            if(init_type is sRGBColor):
                gamut_to_lab = convert_color(sRGBColor(lop[pal_i][color_i][0], lop[pal_i][color_i][1], lop[pal_i][color_i][2]), LabColor, target_illuminant = 'd50')
            elif(init_type is LCHabColor):
                gamut_to_lab = convert_color(LCHabColor(lop[pal_i][color_i][0], lop[pal_i][color_i][1], lop[pal_i][color_i][2]), LabColor, target_illuminant = 'd50')
            elif(init_type is LabColor):
                gamut_to_lab = LabColor(lop[pal_i][color_i][0], lop[pal_i][color_i][1], lop[pal_i][color_i][2])
            if(delta_e_cie2000(WHITE_LAB, gamut_to_lab) > WHITE_THRESH):
                final_pal.append(lop[pal_i][color_i])
        final_lop.append(final_pal)
        final_pal = []
    return final_lop

####################################################################################################
######################################### DATA COLLECTION ##########################################
####################################################################################################
# Load Coloors scraping session
driver = webdriver.Chrome(ChromeDriverManager().install())
driver.get("https://coolors.co/palettes/popular")
time.sleep(5.0)

# Scroll down to get more colors
for i in range(FULL_SCROLL_NUMBER):
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(1.5)
# Check that you have a color div instance in your driver content
try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, COLOR_LOCATION)))
except TimeoutException:
    print('Coloor Scraping: Page timed out after 10 secs. Color-containing DIVS are not showing up for some reason.')
# Store loaded source in BeautifulSoup
soup = BeautifulSoup(driver.page_source)
# Pull all the divs which definitely have colors
condensed = soup.findAll("div", {"class": COLOR_LOCATION})
condensed_STR = [str(div) for div in condensed]
# Extract hex codes from div strings, find all unique values
pattern = re.compile('[a-f0-9A-F]{6}')
COLOORS_HEX_values_o = list(set([tuple(re.findall(pattern, condensed_STR[i])) for i in range(len(condensed_STR))]))

time.sleep(1)

# Adobe Scraping
driver_adobe = webdriver.Chrome(ChromeDriverManager().install())
driver_adobe.get('https://color.adobe.com/explore')
time.sleep(5.0)

# Go past blocks if they exist on page
if(check_exists_by_xpath('//*[@id="onetrust-policy"]', driver_adobe)):
    python_block = driver_adobe.find_element_by_xpath('//*[@id="react-spectrum-134"]/div/div[3]')
    python_block.click()
    python_block = driver_adobe.find_element_by_xpath('//*[@id="onetrust-pc-btn-handler"]')
    python_block.click()
    python_block = driver_adobe.find_element_by_xpath('//*[@id="ot-pc-logo-button"]/button[1]')
    python_block.click()

python_button = driver_adobe.find_element_by_xpath('//*[@id="react-spectrum-8-trigger"]')
python_button.click()
python_color_themes_box = driver_adobe.find_element_by_xpath('/html/body/div[1]/div/div/div[2]/ul/li[2]')
hover = ActionChains(driver_adobe).move_to_element(python_color_themes_box)
hover.perform()
python_button = driver_adobe.find_element_by_xpath('/html/body/div[1]/div/div/div[3]/ul/ul[1]/li[1]/span')
python_button.click()

try:
    WebDriverWait(driver_adobe, 10).until(EC.presence_of_element_located((By.CLASS_NAME, COLOR_LOCATION_ADOBE)))
except TimeoutException:
    print('1 - Adobe Scraping: Page timed out after 10 secs. Color-containing DIVS are not showing up for some reason.')

ALL_ADOBE_DIVS = []
# Access colors on page
for page_num in range(ADOBE_FULL_SCROLL_NUMBER):
    # Scroll down to get more colors
    driver_adobe.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    soup_ADOBE = BeautifulSoup(driver_adobe.page_source)
    condensed_ADOBE = soup_ADOBE.findAll("div", {"class": COLOR_LOCATION_ADOBE})
    condensed_ADOBE_STR = [str(div) for div in condensed_ADOBE]
    ALL_ADOBE_DIVS.append(condensed_ADOBE_STR)
    # Click next page button
    next_page_button = driver_adobe.find_element_by_xpath('//*[@id="color-root"]/div[1]/div[3]/div[3]/button[3]')
    next_page_button.click()
    try:
        WebDriverWait(driver_adobe, 10).until(EC.presence_of_element_located((By.CLASS_NAME, COLOR_LOCATION_ADOBE)))
    except TimeoutException:
        print('2 - Adobe Scraping: Page timed out after 10 secs. Color-containing DIVS are not showing up for some reason.')

ALL_ADOBE_DIVS = list(set(list(chain.from_iterable(ALL_ADOBE_DIVS))))
num_ADOBE_palettes = len(ALL_ADOBE_DIVS)
pattern_ADOBE = re.compile("\\((-?\\d+),(-?\\d+),(-?\\d+)\\)")
ADOBE_HEX_values_o = list(set([tuple([colormap.rgb2hex(int(color_str[0]), int(color_str[1]), int(color_str[2])).replace('#', "") for color_str in re.findall(pattern_ADOBE, ALL_ADOBE_DIVS[i].replace(", ", ","))]) for i in range(num_ADOBE_palettes)]))

HEX_values_o = list(set(COLOORS_HEX_values_o + ADOBE_HEX_values_o))
# len(flat_RBG)
#######################################################################################################
########################################### DATA PROCESSING ###########################################
#######################################################################################################
# Convert from HEX -> RGB (sRGB -> sRGB). No overlap at all (still in sRGB gamut).
RGB_values_o = [tuple([HEX_TO_RGB(HEX_CODE, clamped = True) for HEX_CODE in HEX_PALETTE]) for HEX_PALETTE in HEX_values_o]
# Convert from RGB -> RGB (sRGB [-> CIE XYZ] -> CIE LAB). No overlap between sRGB to CIE LAB directly, but CIE XYZ???.
LAB_values_o = [tuple([convert_color(sRGBColor(RGB_CODE[0], RGB_CODE[1], RGB_CODE[2]), LabColor, target_illuminant = 'd50').get_value_tuple() for RGB_CODE in RGB_PALETTE]) for RGB_PALETTE in RGB_values_o]
# Convert from LAB -> LCH (CIE LAB -> CIE LCH). No overlap between CIE LAB to CIE LCH.
LCH_values_o = [tuple([convert_color(sRGBColor(RGB_CODE[0], RGB_CODE[1], RGB_CODE[2]), LCHabColor, target_illuminant = 'd50').get_value_tuple() for RGB_CODE in RGB_PALETTE]) for RGB_PALETTE in RGB_values_o]

RGB_values = remove_whites(RGB_values_o, sRGBColor)
LAB_values = remove_whites(LAB_values_o, LabColor)
LCH_values = remove_whites(LCH_values_o, LCHabColor)

# Convert color values from list to DataFrame
# flat_HEX_o = ['#' + HEX for HEX in [item for sublist in HEX_values_o for item in sublist]]
flat_RBG = np.array([RGB for RGB in [item for sublist in RGB_values for item in sublist]])
df_RGB = pd.DataFrame([item for sublist in RGB_values for item in sublist]) * 255.0
df_RGB.columns = ['R', 'G', 'B']
df_LAB = pd.DataFrame([item for sublist in LAB_values for item in sublist])
df_LAB.columns = ['L', 'A', 'B']
df_LCH = pd.DataFrame([item for sublist in LCH_values for item in sublist])
df_LCH_ORIGINAL = df_LCH.copy()
df_LCH.columns = ['L', 'C', 'H']
df_LCH.insert(0, 'X', 0.0)
df_LCH.insert(1, 'Y', 0.0)
for index in range(len(df_LCH.index)):
    df_LCH.at[index, 'X'] = df_LCH.at[index, 'C'] * cos(radians(df_LCH.at[index, 'H']))
    df_LCH.at[index, 'Y'] = df_LCH.at[index, 'C'] * sin(radians(df_LCH.at[index, 'H']))
df_LCH.drop(['C', 'H'], 1, inplace = True)
flat_LAB = np.asarray(df_LAB)

#########################################################
#########################################################
# # COLORMIND INTEGRATION
# all_sets = []
# all_pal_sets = []
# existing_palette = ''
# number_colors_total = 5
# first_x = 3
# pal_num_THRESH = 10
# for pal_num in range(len(RGB_values)):
#     incomplete_list = RGB_values[pal_num][:first_x]
#     format = [str(color) for color in [[round(r * 255), round(g * 255), round(b * 255)] for r, g, b in incomplete_list]]
#     final_given = ''
#     final_N = ''
#     for color_i in range(len(format)):
#         if(color_i == len(format) - 1):
#             final_given += format[color_i]
#             break
#         final_given += format[color_i] + ','
#     for N_i in range(5 - first_x):
#         if(N_i == 5 - first_x - 1):
#             final_N += '\"N\"'
#             break
#         final_N += '\"N\",'
#     for n in range(number_colors_total):
#         data = '{"input":[' + final_given + ',' + final_N + '], "model":"default"}'
#         response = ast.literal_eval(requests.post('http://colormind.io/api/', data=data).text)
#         fit_colors = list(chain.from_iterable(response.values()))
#         all_sets.append(fit_colors)
#     all_pal_sets.append(all_sets)
#     if(pal_num == pal_num_THRESH):
#         break
#
# [[display_palette_from_RGB(fit_colors, clamped = False) for fit_colors in all_sets] for all_sets in all_pal_sets]

#######################################################################################################
########################################### DATA ANALYTICS ############################################
#######################################################################################################
#########################################################
#########################################################
# Plot 3D Scatterplots of Colours in different Gamuts
ax_sRGB_to_sRGB = plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in sRGB Gamut', ('R', 'G', 'B'), ((0, 255), (0, 255), (0, 255)), df_RGB, flat_RBG, False, None)
ax_sRGB_to_CIELAB = plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in CIELAB Gamut', ('A', 'B', 'L'), ((-200, 200), (-200, 200), (0, 100)), df_LAB, flat_RBG, False, None)
ax_sRGB_to_CIELCH = plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in CIELCH Gamut', ('C', 'C', 'L'), ((-200, 200), (-200, 200), (0, 100)), df_LCH, flat_RBG, True, ('X', 'Y', 'L'))
#########################################################
#########################################################
# Execute all PCA processes for all Gamuts
pca_sRGB, kmeans_PCA_sRGB, X_Fit_sRGB_to_PCA, y_kmeans_PCA_sRGB, ax_PCA_sRGB = complete_PCA_process('sRGB Gamut Clustered in PCA Space', df_RGB)
pca_LAB, kmeans_PCA_LAB, X_Fit_CIELAB_to_PCA, y_kmeans_PCA_LAB, ax_PCA_CIELAB = complete_PCA_process('CIELAB Gamut Clustered in PCA Space', df_LAB)
pca_LCH, kmeans_PCA_LCH, X_Fit_CIELCH_to_PCA, y_kmeans_PCA_LCH, ax_PCA_CIELCH = complete_PCA_process('CIELCH Gamut Clustered in PCA Space', df_LCH_ORIGINAL)
#########################################################
#########################################################
ax_RGB_inverse_transform_CENTROID_PLOTS, df_RGB_pca_sRGB, sRGB_plot_ORIGINAL = display_inv_transform_sRGB('sRGB-PCA-sRGB', pca_sRGB, kmeans_PCA_sRGB, X_Fit_sRGB_to_PCA, y_kmeans_PCA_sRGB, RGB_values, plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in sRGB Gamut (w/ sRGB-PCA Centroid Connections)', ('R', 'G', 'B'), ((0, 255), (0, 255), (0, 255)), df_RGB, flat_RBG, False, None))
ax_LAB_inverse_transform_CENTROID_PLOTS, df_LAB_pca_sRGB, sRGB_plot_ORIGINAL = display_inv_transform_sRGB('LAB-PCA-sRGB', pca_LAB, kmeans_PCA_LAB, X_Fit_CIELAB_to_PCA, y_kmeans_PCA_LAB, RGB_values, plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in sRGB Gamut (w/ LAB-PCA Centroid Connections)', ('R', 'G', 'B'), ((0, 255), (0, 255), (0, 255)), df_RGB, flat_RBG, False, None))
ax_LCH_inverse_transform_CENTROID_PLOTS, df_LCH_pca_sRGB, sRGB_plot_ORIGINAL = display_inv_transform_sRGB('LCH-PCA-sRGB', pca_LCH, kmeans_PCA_LCH, X_Fit_CIELCH_to_PCA, y_kmeans_PCA_LCH, RGB_values, plot3D_object(GLOBAL_FIG_SIZE, 'Coolors sRGB Gamut in sRGB Gamut (w/ LCH-PCA Centroid Connections)', ('R', 'G', 'B'), ((0, 255), (0, 255), (0, 255)), df_RGB, flat_RBG, False, None))
#########################################################
#########################################################
# Annotate binned HEX List with group numbers
ALL_RGB_from_sRGB_annotations = [[color_location(df_RGB_pca_sRGB, [i * 255.0 for i in color]) for color in palette] for palette in RGB_values]
ALL_RGB_from_LAB_annotations = [[color_location(df_LAB_pca_sRGB, [i * 255.0 for i in color]) for color in palette] for palette in RGB_values]
ALL_RGB_from_LCH_annotations = [[color_location(df_LCH_pca_sRGB, [i * 255.0 for i in color]) for color in palette] for palette in RGB_values]

# # Make Affinity Net (based on properties of clusters)
# for split_threshold in TOP_CONST:
#     populate_files_ML(OPEN_TAB, ALL_RGB_from_sRGB_annotations, K, split_threshold, VIS_parameters, shape_name, VIS_location, prim_dir_name = 'sRGB-sRGB-PCA-sRGB/')
#     populate_files_ML(OPEN_TAB, ALL_RGB_from_LAB_annotations, K, split_threshold, VIS_parameters, shape_name, VIS_location, prim_dir_name = 'sRGB-LAB-PCA-sRGB/')
#     populate_files_ML(OPEN_TAB, ALL_RGB_from_LCH_annotations, K, split_threshold, VIS_parameters, shape_name, VIS_location, prim_dir_name = 'sRGB-LCH-PCA-sRGB/')

#########################################################
########## NAVIGATE AFFINITY NET TO FIND COLORS #########
#########################################################


# nx.draw(LAB_G)

#####################
# COMBINATIONS (white not exclued yet, MAX_CENTROID_ACCESS = 3) #
# DIVERSITY = False, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'HISTORICAL MAX' --> BAD: Lots of white, some very purple
# DIVERSITY = False, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'MAX'            --> BAD: Lots of white, lots of oscillation
# DIVERSITY = False, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'HISTORICAL MAX'         --> OKAY: A little more muted, lots of low-L colors
# DIVERSITY = False, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'MAX'                    --> BETTER THAN LAST: A little less muted than last, some low-L colors
# DIVERSITY = True, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'HISTORICAL MAX'  --> Lots of oscillation
# DIVERSITY = True, COLOR_PICK_METHOD = 'REPRESENTATIVE', CENTROID_PICK_METHOD = 'MAX'             --> Lots of oscillation
# DIVERSITY = True, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'HISTORICAL MAX'          --> BETTER: Some whites, even dist of L colors
# DIVERSITY = True, COLOR_PICK_METHOD = 'RANDOM', CENTROID_PICK_METHOD = 'MAX'                     --> BETTER THAN LAST: Fewer whites, even dist of L colors, more L than before
DIVERSITY = True
if(DIVERSITY):
    MAX_CENTROID_ACCESS = 1
else:
    MAX_CENTROID_ACCESS = None
COLOR_PICK_METHOD = 'RANDOM'
CENTROID_PICK_METHOD = 'MAX'
TRUNCATE_START = 5
NUM_COLORS = 15
NUM_PALETTES = 15

start_pick = random.choice([palette for palette in RGB_values if len(palette) == 5])[:TRUNCATE_START]
display_palette_from_RGB(start_pick)
LAB_color_maps = []
# time.sleep(2)
for palette_num in range(NUM_PALETTES):
    RGB_centroid_list = list(set(df_RGB_pca_sRGB['Category']))
    sRGB_G = graph_from_weights(ALL_RGB_from_sRGB_annotations)
    LAB_centroid_list = list(set(df_LAB_pca_sRGB['Category']))
    LAB_G = graph_from_weights(ALL_RGB_from_LAB_annotations)
    LCH_centroid_list = list(set(df_LCH_pca_sRGB['Category']))
    LCH_G = graph_from_weights(ALL_RGB_from_LCH_annotations)

    traverse = True
    outer_iteration = 1     # Must be 1, cannot change
    chosen_clusters = []
    # ('Traversal Iteration', 'Cluster ID', 'Weight of Self-loop', 'Number of Top Repeats')
    self_loops = []
    LAB_color_map = []
    # ('Color', 'Delta-E from Centroid', 'Outer Iteration')
    LAB_color_map_detailed = []
    delta_E_tracker = []
    sRGB_CENTROIDS = pca_sRGB.inverse_transform(kmeans_PCA_sRGB.cluster_centers_)
    LAB_CENTROIDS = pca_LAB.inverse_transform(kmeans_PCA_LAB.cluster_centers_)
    LCH_CENTROIDS = pca_LCH.inverse_transform(kmeans_PCA_LCH.cluster_centers_)

    dummy_c = 1
    random.shuffle(start_pick)
    for color in start_pick:
        centroid = int(color_location(df_LAB_pca_sRGB, np.asarray(color) * 255.0))
        chosen_clusters.append(centroid)

        centroid_color_OBJ = LabColor(LAB_CENTROIDS[centroid][0], LAB_CENTROIDS[centroid][1], LAB_CENTROIDS[centroid][2])
        cluster_color_OBJ = convert_color(sRGBColor(color[0], color[1], color[2]), LabColor, target_illuminant = 'd50')
        delta_E = delta_e_cie2000(centroid_color_OBJ, cluster_color_OBJ)
        delta_E_tracker.append(delta_E)

        current_color_LAB_OBJ = convert_color(sRGBColor(color[0], color[1], color[2]), LabColor, target_illuminant = 'd50')
        LAB_color_map.append(current_color_LAB_OBJ)
        LAB_color_map_detailed.append((current_color_LAB_OBJ, centroid, dummy_c))
        dummy_c += 1
        next_cluster = centroid
    outer_iteration = TRUNCATE_START + 1

    while(traverse == True):
        ###################################################
        ################# PICK CENTROID* ##################         # WORKS AS EXPECTED
        if(outer_iteration == 1):
            # Pick a random centroid cluster
            current_cluster = int(random.choice(LAB_centroid_list))
        else:
            current_cluster = next_cluster
            next_cluster = None
        chosen_clusters.append(current_cluster)
        ###################################################
        ################# PICK COLOR ######################          # DOES NOT WORK AS EXPECTED, COLOR CONVERSION OUT OF BOUNDS
        # Configure object for Centroid in Cluster
        centroid_color_OBJ = LabColor(LAB_CENTROIDS[current_cluster][0], LAB_CENTROIDS[current_cluster][1], LAB_CENTROIDS[current_cluster][2])
        # Determine all colors in centroid
        cluster_colors = np.asarray(df_LAB_pca_sRGB.loc[df_LAB_pca_sRGB['Category'] == current_cluster].reset_index().drop(['index', 'Category'], 1))
        # Configure LabColor object for all Centroid-specific cluster colors
        LAB_LIST_OBJ = [convert_color(sRGBColor(RGB_CODE[0]/255.0, RGB_CODE[1]/255.0, RGB_CODE[2]/255.0), LabColor, target_illuminant = 'd50') for RGB_CODE in cluster_colors]
        # Filter LIST_OBJ (all the colors in the specific centroid) to not include previously chosen colors (non-replacement)
        # Unintended (but not harmful) consequence: may delete all repeats, even if same color is repeated in LIST_OBJ
        LAB_LIST_OBJ = remove_occurences_LAB(LAB_LIST_OBJ, LAB_color_map)
        if(len(LAB_LIST_OBJ) == 0):
            print('TRAVERSAL CHECKPOINT: No more data in cluster ' + str(current_cluster) + ' to work with. outer_iteration: ' + str(outer_iteration))
            LAB_centroid_list.remove(current_cluster)
            LAB_G.remove_node(current_cluster)
            next_cluster = int(random.choice(LAB_centroid_list))
            continue

        # Pick color in current centroid based on preference in the space
        if(COLOR_PICK_METHOD == 'RANDOM'):
            # Randomly pick a color within the centroid
            cluster_color_OBJ = random.choice(LAB_LIST_OBJ)
            delta_E = delta_e_cie2000(centroid_color_OBJ, cluster_color_OBJ)
            delta_E_tracker.append(delta_E)
        elif(COLOR_PICK_METHOD == 'REPRESENTATIVE'):
            # Pick color that is most "similar" to the centroid color
            # Associates delta-e for each color in centroid with the centroid as the fixed reference point
            cluster_colors_delta_E_match = [delta_e_cie2000(centroid_color_OBJ, color) for color in LAB_LIST_OBJ]
            color_delta_e = list(zip(LAB_LIST_OBJ, cluster_colors_delta_E_match))
            color_delta_e = sorted(color_delta_e, key = lambda tup: tup[1])
            cluster_color_OBJ = color_delta_e[0][0]
            delta_E_tracker.append(color_delta_e[0][1])
        else:
            print('FATAL ERROR: \'PICK_METHOD\' improperly initialized')

        LAB_color_map.append(cluster_color_OBJ)
        LAB_color_map_detailed.append((cluster_color_OBJ, current_cluster, outer_iteration))

        ###################################################
        ################# PICK NEXT CENTROID ##############         # PENDING REVIEW
        # Determine number of neighbours of selected node/centroid (from list of neighbours for every node/centroid)
        # neighbors = sorted([(node, len(set(list(LAB_G.neighbors(node))))) for node in LAB_G], key = lambda tup: tup[0])[int(current_cluster)][1]
        # Determine weights for each edge with chosen node
        connected_clusters = [tup[1] for tup in list(LAB_G.edges(int(current_cluster)))]
        connected_weights = sorted([(connected_cluster, LAB_G[int(current_cluster)][connected_cluster]['weight']) for connected_cluster in connected_clusters], key = lambda tup: tup[1], reverse = True)

        top_weight = -1     # dummy assignment, must be initially set as num <= 0
        top_clusters = []   # Tracks centroid IDs with highest weight
        counter = 1         # Local for-loop counter for debugging purposes
        # Determines highest weight
        for connection in connected_weights:
            if(connection[1] > top_weight):
                top_weight = connection[1]
                top_clusters.append(connection[0])
            elif(connection[1] == top_weight):
                top_clusters.append(connection[0])
            elif(connection[1] < top_weight):
                if(counter == 0):
                    print('FATAL ERROR: \'connected_weights\' has a negative weight')
                    break
                else:
                    print('SUCCESS: Highest weight found after iteration ' + str(counter))
                    break
            counter += 1
        # Store the second best cluster though (so len(list) != 0 if the only cluster in top_clusters is a self_loop)
        top_clusters.append(connection[0])
        # Check for self loops and possible delete based on DIVERSITY
        if(current_cluster in top_clusters):
            # Self loop has occurred, which is fine, but track it.
            self_loops.append((outer_iteration, current_cluster, top_weight, len(top_clusters)))
            if(DIVERSITY == True):
                top_clusters.remove(current_cluster)

        if(CENTROID_PICK_METHOD == 'MAX'):
            next_cluster = int(random.choice(top_clusters))
        elif(CENTROID_PICK_METHOD == 'HISTORICAL MAX'):
            # Check how many times current_cluster has been visited
            historical_clusters = dict(collections.Counter([cluster for d_color, cluster, d_iteration in LAB_color_map_detailed]))
            # Decide if self-loop should be excluded (add gradient hyperparameter?)
            if(DIVERSITY):
                # Check if there's a self loop and delete current_cluster from possible options if been accessed over MAX_CENTROID_ACCESS times
                try: # there is a self-loop, remove self-loop and any future option with > 3 traversals (consec. or non-consec.)
                    tuple_of_current_cluster_connection = [relation for relation in connected_weights if relation[0] == current_cluster][0]
                    if(historical_clusters[current_cluster] >= MAX_CENTROID_ACCESS):
                        connected_weights.remove(tuple_of_current_cluster_connection)   # Remove self-loop
                        proposed_next_cluster = int(connected_weights[0][0])            # Next expected color
                        proposed_next_cluster_tuple = [relation for relation in connected_weights if relation[0] == proposed_next_cluster][0]   # expected tuple
                        try:
                            while(historical_clusters[proposed_next_cluster] > MAX_CENTROID_ACCESS):    # if proposed_next_cluster access > MAX_CENTROID_ACCESS
                                connected_weights.remove(proposed_next_cluster_tuple)                   # remove from candidates
                                proposed_next_cluster = int(connected_weights[0][0])                    # re-assign proposed next cluster
                        except:
                            pass # proposed_next_cluster is not in collections, meaning it hasn't been accessed (it will be set as the next_cluster)
                except IndexError: # there isn't a self-loop
                    proposed_next_cluster = int(connected_weights[0][0])                # Next expected color
                    proposed_next_cluster_tuple = [relation for relation in connected_weights if relation[0] == proposed_next_cluster][0]       # expected tuple
                    try:
                        while(historical_clusters[proposed_next_cluster] > MAX_CENTROID_ACCESS):        # if proposed_next_cluster access > MAX_CENTROID_ACCESS
                            connected_weights.remove(proposed_next_cluster_tuple)                       # remove from candidates
                            proposed_next_cluster = int(connected_weights[0][0])                        # re-assign proposed next cluster
                    except:
                        pass # proposed_next_cluster is not in collections, meaning it hasn't been accessed (it will be set as the next_cluster)
            # Decide what other centroids should be excluded to prevent oscillation among n-few centroids (analyze historical data)
            # Assign/set node with maximum weight as next_cluster (could be same as current_centroid depending on DIVERSITY)
            next_cluster = int(connected_weights[0][0])
        else:
            print('FATAL ERROR: \'CENTROID_PICK_METHOD\' improperly initialized')

        # If no more data anywhere to work with OR request number threshold has been reached
        if(outer_iteration == NUM_COLORS or len(LAB_centroid_list) == 0):
            traverse = False
        else:
            outer_iteration += 1
        top_clusters = []
    LAB_color_maps.append(LAB_color_map)
LAB_RGB_color_maps = [[convert_color(LAB_CODE, sRGBColor, target_illuminant = 'd50').get_value_tuple() for LAB_CODE in LAB_color_map] for LAB_color_map in LAB_color_maps]
# [colormap.rgb2hex(round(r), round(g), round(b)) for r, g, b, in LAB_RGB_color_map]
[display_palette_from_RGB(LAB_RGB_color_map, clamped = True) for LAB_RGB_color_map in LAB_RGB_color_maps]
# sns.set_palette(distinct_colors)
# sns.palplot(sns.color_palette())

################
# THINGS TO DO #
# 1. Add additional weights between adjacent colors (look good together), test
# 2. Bin Data, test
# 3. Scrape more data
################

plt.show()
plt.close()




#
