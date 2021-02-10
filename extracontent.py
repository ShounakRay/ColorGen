# # page_object = requests.get(url)
# #
# # if(page_object.status_code == 200):
# #     print('Page Successfully Retrieved')
# # else:
# #     print('Page Unsuccessfully Retrieved')
#
# page_content = page_object.content
# children = list(soup.children)
# children_types = [str(type(item)) for item in children]
#
# print(soup.prettify())
#
# # Can only use .index method since their is only one instance of .TAG in children/children_types
# TAG_content = children[children_types.index("<class 'bs4.element.Tag'>")]
# TAG_children = list(TAG_content.children)
# TAG_children_types = [str(type(item)) for item in TAG_children]
#
# print(TAG_content.prettify())
#
# BODY_content = list(TAG_content)[2]
# COLORS = BODY_content.find("div", {"id": "explore-palette_colors"})
# COLORS_children = list(COLORS.children)
# COLORS_children_types = [str(type(item)) for item in COLORS_children]
# len(COLORS)
# # Last child node (no info about colors) is *truly* deleted
# print(COLORS.prettify())
# list(COLORS)[0]
# COLORS = BODY_content.find("div", {"id": "explore-palettes_results"})


############################################################
############################################################
# Determine Optimal k-value for k-means clustering
# inertia_cost = []
# group_freq = []
# for k in range(1, K_MAX + 1):
#     k_means_PCA = KMeans(n_clusters = k, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 3)
#     k_means_PCA.fit(X_Demo_fit_pca)
#     inertia_cost.append([k, k_means_PCA.inertia_])
#     # inter_freq_dict = dict(collections.OrderedDict(sorted(dict(Counter(k_means_PCA.labels_)).items())))
#     # if(k < K_MAX):
#     #     [inter_freq_dict.update({i: None}) for i in range(k, K_MAX)]
#     # group_freq.append([k, inter_freq_dict])
# inertia_cost = pd.DataFrame(inertia_cost)
# inertia_cost.columns = ['K', 'Normalized Inertia (Cost)']
# inertia_cost['Normalized Inertia (Cost)'] = inertia_cost['Normalized Inertia (Cost)'].apply(lambda x: x / max(inertia_cost['Normalized Inertia (Cost)']))
# [inertia_cost[group_num].apply(lambda x: x / max(inertia_cost[group_num])) for group_num in range(2, 32)]
# [inertia_cost.insert(len(inertia_cost.columns), 'Group ' + str(i), list(group_freq[i][1].values())) for i in range(len(group_freq))]
# for group_num in range(2, 32):
# curr_group = inertia_cost[inertia_cost.columns[group_num]]
# for i in range(len(curr_group)):
#     if(math.isnan(curr_group[i]) == False):
#         inertia_cost.at[i, inertia_cost.columns[group_num]] = inertia_cost.at[i, inertia_cost.columns[group_num]]/float(max(curr_group))
# ax_COST = sns.lineplot(x = 'K', y = 'Normalized Inertia (Cost)', data = inertia_cost)
# ax_COST.set_title('Optimize k-value in Clustering')
# ax_COST.set_xticks(range(1, K_MAX + 1))
# ax_COST.set_yticks([x / 10.0 for x in range(0, 11)])
# ax_COST.grid(True)
############################################################
############################################################

############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
# # Plotting all the inverse transformed points in sRGB Space
# fig1 = plt.figure(figsize = (GLOBAL_FIG_SIZE, GLOBAL_FIG_SIZE))
# ax_RGB_inverse_transform = fig1.add_subplot(111, projection = '3d')
# ax_RGB_inverse_transform.set_title('sRGB-PCA-sRGB Inverse Transformed Points in sRGB Space w/ Clustering and Palette Connections')
# ax_RGB_inverse_transform.set_xlabel('R')
# ax_RGB_inverse_transform.set_ylabel('G')
# ax_RGB_inverse_transform.set_zlabel('B')
# ax_RGB_inverse_transform.scatter(df_RGB_pca_sRGB['R'], df_RGB_pca_sRGB['G'], df_RGB_pca_sRGB['B'],
#             c = df_RGB_pca_sRGB['Category'], cmap = 'viridis',
#             edgecolor = 'k', s = 60, alpha = 0.5)
# # Plotting all the inverse transormed cluster centroids in sRGB Space
# ax_RGB_inverse_transform.scatter(df_RGB_CENT['R'], df_RGB_CENT['G'], df_RGB_CENT['B'], s = 300, c = 'r', marker = '*', label = 'Centroid')
# # Connect the points in each palette in sRGB Space
# c = 0 # For color formatting, and *connection limit
# for palette_colors in RGB_values:
#     x = [255.0 * tuple_i[0] for tuple_i in palette_colors]
#     y = [255.0 * tuple_i[1] for tuple_i in palette_colors]
#     z = [255.0 * tuple_i[2] for tuple_i in palette_colors]
#     ax_RGB_inverse_transform.plot(x, y, z, color = distinct_colors[c % len(distinct_colors)])
#     # if(c == 3):
#     #     break
#     c += 1
############################################################
############################################################

# # Inverse Transform All Points (from all previous Gamuts->PCA) back to sRGB Gamut
# df_RGB_pca_sRGB = pd.DataFrame(pca_sRGB.inverse_transform(X_Fit_sRGB_to_PCA.tolist()).tolist())
# df_RGB_pca_sRGB.columns = ['R', 'G', 'B']
# df_RGB_pca_sRGB.insert(3, 'Category', y_kmeans_PCA_sRGB.tolist())
#
# # Inverse Transform Centroid Points to sRGB Gamut
# original_CENTROIDS = pca_sRGB.inverse_transform(kmeans_PCA_sRGB.cluster_centers_.tolist()).tolist()
# df_RGB_pca_sRGB_CENT = pd.DataFrame(original_CENTROIDS)
# df_RGB_pca_sRGB_CENT.columns = ['R', 'G', 'B']

#########################################################
#########################################################
# # Connect the associated centroid of the points in each palette in sRGB Space
# fig_centr = plt.figure(figsize = (GLOBAL_FIG_SIZE, GLOBAL_FIG_SIZE))
# ax_RGB_inverse_transform_CENTROID_PLOTS = fig_centr.add_subplot(111, projection = '3d')
# ax_RGB_inverse_transform_CENTROID_PLOTS.set_title('sRGB-PCA-sRGB Inverse Transformed Points in sRGB Space w/ Clustering and Centroid Palette Connections')
# ax_RGB_inverse_transform_CENTROID_PLOTS.set_xlabel('R')
# ax_RGB_inverse_transform_CENTROID_PLOTS.set_ylabel('G')
# ax_RGB_inverse_transform_CENTROID_PLOTS.set_zlabel('B')
# ax_RGB_inverse_transform_CENTROID_PLOTS.scatter(df_RGB_pca_sRGB['R'], df_RGB_pca_sRGB['G'], df_RGB_pca_sRGB['B'],
#             c = df_RGB_pca_sRGB['Category'], cmap = 'viridis',
#             edgecolor = 'k', s = 60, alpha = 0.5)

# Next two operations (point scatter, centroid plot) are also performed on ax_sRGB_to_sRGB object to reflect centroid calculations visually
# Plotting all the inverse transormed cluster centroids in sRGB Space
# ax_RGB_inverse_transform_CENTROID_PLOTS.scatter(df_RGB_pca_sRGB_CENT['R'], df_RGB_pca_sRGB_CENT['G'], df_RGB_pca_sRGB_CENT['B'], s = 300, c = 'r', marker = '*', label = 'Centroid')
# ax_sRGB_to_sRGB.scatter(df_RGB_pca_sRGB_CENT['R'], df_RGB_pca_sRGB_CENT['G'], df_RGB_pca_sRGB_CENT['B'], s = 300, c = 'r', marker = '*', label = 'Centroid')
# Connect the points in each palette in sRGB Space (based on respective centroids)
# c = 0 # For color formatting, and *connection limit
# for palette_colors in RGB_values:
#     loc = [[color_location(df_RGB_pca_sRGB, tuple([[i * 255.0 for i in color][0], [i * 255.0 for i in color][1], [i * 255.0 for i in color][2]]))] for color in palette_colors]
#     loc = list(chain.from_iterable(loc))
#     CENT_points = [list(chain.from_iterable(df_RGB_pca_sRGB_CENT.iloc[[loc_i]].to_numpy().tolist())) for loc_i in loc]
#     CENT_points_x = [val[0] for val in CENT_points]
#     CENT_points_y = [val[1] for val in CENT_points]
#     CENT_points_z = [val[2] for val in CENT_points]
#     [ax_RGB_inverse_transform_CENTROID_PLOTS.plot(CENT_points_x, CENT_points_y, CENT_points_z, color = distinct_colors[c % len(distinct_colors)])]
#      ax_sRGB_to_sRGB.plot(CENT_points_x, CENT_points_y, CENT_points_z, color = distinct_colors[c % len(distinct_colors)])
#     c += 1
############################################################
############################################################
############################################################
############################################################
############################################################
############################################################
