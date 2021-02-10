import itertools
import collections
import pandas as pd
import networkx as nx
from itertools import chain
import random
import colorsys
import seaborn as sns

GLOBAL_DISTINCT_COLORS = ["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059", "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87", "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80", "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100", "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F", "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09", "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66", "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C", "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81", "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00", "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700", "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329", "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C", "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800", "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51", "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58", "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D", "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176", "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5", "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4", "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01", "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966", "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0", "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C", "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868", "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183", "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433", "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F", "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E", "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F", "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00", "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66", "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25", "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]

# Round to dec decimal places, properly
def proper_round(num, dec=0):
    num = str(num)[:str(num).index('.')+dec+2]
    if num[-1]>='5':
        return float(num[:-2-(not dec)]+str(int(num[-2-(not dec)])+1))
    return float(num[:-1])

# color_map not specified as argument, but created in graph_preproccessing
def list_clustering_graph_to_json(Description, K, top_CONST, VIS_parameters, shape_name, dict_stratClass_edgeClass, d_keys_ind, d_values_ind, edge_weights_3t, color_strat_dict, VIS_location = '', init_strat_name = ''):
    # RENDERING TIME IS EXTREMELY SLOW: POSSIBLY NORMALIZE THE WEIGHTS (LN 457), LOOKS VERY MESSY (ASSESS REPULSION AND PHYSICS, STABILIZATION METRICS (DECREASE -> FASTER?))
    # GENERATE SPECIALIZED JSON FILE
    HTML_FILE = """"""
    HTML_FILE += """
    <html>
        <head>
            <script type="text/javascript" src=\""""
    HTML_FILE += VIS_location + """vis/dist/vis.js"></script>"""
    HTML_FILE += """
            <link href=""" + VIS_location + """vis/dist/vis.css rel="stylesheet" type="text/css" />
            <style type="text/css">
                #mynetwork
                {
                    width: 100%;
                    height: 90%;
                    border: 3px solid lightgray;
                }
            </style>
        </head>"""
    HTML_FILE += """
        <body>
            <div id = "mynetwork"></div>"""
    if(VIS_parameters[0]): # buttons
        HTML_FILE += """
            <p>These are different clustering methods (functions)</p>
            <input type = "button" onclick = "clusterByConnection()" value = "Cluster 'None' by connections">
            <br>
            <input type = "button" onclick = "clusterByHubsize()" value = "Cluster by hubsize">
            <br>
            <input type = "button" onclick = "clusterByColor()" value = "Cluster by color">"""
    if(VIS_parameters[10]):
        HTML_FILE += """
            <h2 style = "text-align:center;font-family:Helvetica;font-size:1.5em; margin-bottom: 2px;">"""
        HTML_FILE += str(init_strat_name) + Description + ' Color Centroid Interaction' + "; Top " + str(top_CONST) + " Links</h2>"
    if(VIS_parameters[2]):
        HTML_FILE += """
            <div class='my-legend'>
                <div class='legend-scale'>
                    <ul class='legend-labels'>
                        """
        for num in range(len(color_strat_dict)):
            HTML_FILE += """<li><span style='background:""" + str(list(color_strat_dict.values())[num]) + """;'></span>""" + str(list(color_strat_dict.keys())[num]) + """</li>
                        """
        HTML_FILE += """
                </ul>
            </div>
        </div>
        <style type='text/css'>
          .my-legend {
            display: table;
            margin: 0 auto;
            margin-top: 0.5em;
            }
          .my-legend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
            }
          .my-legend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
            }
          .my-legend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            line-height: 18px;
            margin-bottom: 5px;
            display: inline;
            float: left;
            margin-left: 1em;
            }
          .my-legend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 1px solid #999;
            }
          .my-legend .legend-source {
            font-size: 70%;
            color: #999;
            clear: both;
            }
          .my-legend a {
            color: #777;
            }
        </style>
        """
    # HTML_FILE += """
    #         <div id = "mynetwork"></div>"""
    HTML_FILE += """
            <script type="text/javascript">"""

    # Loop for nodes, edges, options, functions (ONE+ INDENTS)
    HTML_FILE += """
                var nodes = new vis.DataSet([\n"""
    for node_num in range(len(dict_stratClass_edgeClass)):
        HTML_FILE += """                {"id": """ + str(node_num)
        if(VIS_parameters[4]): # node labels
            HTML_FILE += """, "label": """ + "\"" + str(d_keys_ind[node_num]) + "\""
        if(VIS_parameters[1]): # groups attribute
            HTML_FILE += """, "group": """ + "\"" + str(d_values_ind[node_num]) + "\""
        if(VIS_parameters[8]): # shapes
            HTML_FILE += """, "shape": """ + "\"" + str(shape_name) + "\""
        color_faded = [item + 'F2' for item in GLOBAL_DISTINCT_COLORS[-K:]][node_num]
        HTML_FILE += """, color: {'border':'""" + color_faded + """', 'highlight':'""" + color_faded + "', " + "'hover':'" + color_faded + "'" + """}"""
        # if(VIS_parameters[9]): # node values/sizes
            # HTML_FILE += """, "value": """ + "\"" + someListUndefined + "\""
        if(node_num <= len(dict_stratClass_edgeClass) - 2):
            HTML_FILE += """},\n"""
        else:
            HTML_FILE += """}\n"""
    HTML_FILE += """            ]);"""
    HTML_FILE += """
                var edges = new vis.DataSet([\n"""
    c = 0
    for edge in edge_weights_3t:
        HTML_FILE += """                {"from": """ + str(d_keys_ind.index(edge[0]))
        HTML_FILE += """, "to": """ + str(d_keys_ind.index(edge[1]))
        HTML_FILE += """, "value": """ + str(edge[2])
        HTML_FILE += """, "color": {'inherit': 'both'}"""
        # if(edge_labels):
            # do something but there's no code here since what in the world would be the label (maybe the weight?)
        if(c <= len(edge_weights_3t) - 2):
            HTML_FILE += """},\n"""
        else:
            HTML_FILE += """}\n"""
        c += 1
    HTML_FILE += """            ]);"""
    HTML_FILE+="""
                var container = document.getElementById('mynetwork');
                var data = {
                    nodes: nodes,
                    edges: edges
                };"""
    HTML_FILE += """
                var options = {"""
    if(VIS_parameters[2]):
        HTML_FILE += """
                    groups: {\n"""
        for strat_num in range(len(list(color_strat_dict.keys()))):
            HTML_FILE += """\t\t\t\t\t\t\t\t\t\'""" + str(list(color_strat_dict.keys())[strat_num]) + "\': {color: {background: """ + "\"" + list(color_strat_dict.values())[strat_num] + """\"}, borderWidth: 1"""
            if(strat_num <= len(list(color_strat_dict.keys())) - 2):
                HTML_FILE += """},\n"""
            else:
                HTML_FILE += """}\n"""
        HTML_FILE += """                }"""
    if(VIS_parameters[3]): #physics option
        if(VIS_parameters[2]): HTML_FILE += "," #groups option
        HTML_FILE += """
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "theta": 0.5,
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.3,
                            "springLength": 95,
                            "springConstant": 0.04,
                            "damping": 0.09,
                            "avoidOverlap": 0.1
                        },
                        "forceAtlas2Based": {
                            "theta": 0.5,
                            "gravitationalConstant": -50,
                            "centralGravity": 0.020,
                            "springConstant": 0.01,
                            "springLength": 100,
                            "damping": 0.4,
                            "avoidOverlap": 0.1
                        },
                        "repulsion": {
                            "centralGravity": 0.2,
                            "springLength": 200,
                            "springConstant": 0.05,
                            "nodeDistance": 100,
                            "damping": 0.09
                        },
                        "maxVelocity": 50,
                        "minVelocity": 0.1,
                        "solver": 'forceAtlas2Based',
                        "stabilization": {
                          "enabled": true,
                          "iterations": 1000,
                          "updateInterval": 100,
                          "onlyDynamicEdges": false,
                          "fit": true
                        },
                        "adaptiveTimestep": true
                    }"""
    if(VIS_parameters[5]): #manipulation option
        if(VIS_parameters[3]): HTML_FILE += "," #physics option
        HTML_FILE += """
                    "manipulation": {
                        "enabled": true,
                        "initiallyActive":false,
                        "addNode":true,
                        "addEdge":true,
                        "editNode":undefined,
                        "editEdge":true
                    }"""
    if(VIS_parameters[6]): #interaction option
        if(VIS_parameters[5]): HTML_FILE += "," #manipulation option
        HTML_FILE += """
                    "interaction": {
                        "dragNodes":true,
                        "dragView": true,
                        "hideEdgesOnDrag": false,
                        "hideEdgesOnZoom": true,
                        "hideNodesOnDrag": false,
                        "hover": true,
                        "hoverConnectedEdges": true,
                        "multiselect": true,
                        "navigationButtons": false,
                        "selectable": true,
                        "selectConnectedEdges": true,
                        "tooltipDelay": 0,
                        "zoomView": true
                    }"""
    if(VIS_parameters[7]): #layout option
        if(VIS_parameters[6]): HTML_FILE += "," #interaction option
        HTML_FILE += """
                    "layout": {
                        "improvedLayout": true,
                        "clusterThreshold":250
                    }"""
    HTML_FILE += """
                };"""
    HTML_FILE += """
                var network = new vis.Network(container, data, options)

                network.on("selectNode", function(params)
                {
                    if (params.nodes.length == 1)
                    {
                        if (network.isCluster(params.nodes[0]) == true)
                        {
                            network.openCluster(params.nodes[0]);
                        }
                    }
                });"""
    if(VIS_parameters[0]):
        HTML_FILE += """
                function clusterByConnection()
                {
                    network.setData(data);
                    network.clusterByConnection(1)
                }
                function clusterByHubsize()
                {
                    network.setData(data);
                    var clusterOptionsByData =
                    {
                        processProperties: function(clusterOptions, childNodes)
                        {
                            clusterOptions.label = "<" + childNodes.length + ">";
                            return clusterOptions;
                        },
                        clusterNodeProperties: {borderWidth:4, shape:'database', font:{size:29}}
                    };
                    network.clusterByHubsize(undefined, clusterOptionsByData);
                }
                function clusterByColor()
                {
                    network.setData(data);
                    var colors = ['blue','pink','green'];
                    var clusterOptionsByData;
                    for (var i = 0; i < colors.length; i++)
                    {
                        var color = colors[i];
                        clusterOptionsByData =
                        {
                            joinCondition: function (childOptions)
                            {
                            return childOptions.color.background == color;
                            },
                            processProperties: function (clusterOptions, childNodes, childEdges)
                            {
                                var totalMass = 0;
                                for (var i = 0; i < childNodes.length; i++)
                                {
                                    totalMass += childNodes[i].mass;
                                }
                                clusterOptions.mass = totalMass;
                                return clusterOptions;
                            },
                            clusterNodeProperties: {id: 'cluster:' + color, borderWidth: 3, shape: 'database', color:color, label:'color:' + color}
                        };
                        network.cluster(clusterOptionsByData);
                    }
                }"""
    HTML_FILE += """
            </script>
        </body>
    </html>"""

    return HTML_FILE

# color_map not specified as argument, but created in graph_preproccessing
def graph_to_json(edge_class, strat_class, top_CONST, VIS_parameters, shape_name, dict_stratClass_edgeClass, d_keys_ind, d_values_ind, edge_weights_3t, color_strat_dict, VIS_location = '', init_strat_name = ''):
    # RENDERING TIME IS EXTREMELY SLOW: POSSIBLY NORMALIZE THE WEIGHTS (LN 457), LOOKS VERY MESSY (ASSESS REPULSION AND PHYSICS, STABILIZATION METRICS (DECREASE -> FASTER?))
    # GENERATE SPECIALIZED JSON FILE
    HTML_FILE = """"""
    HTML_FILE += """
    <html>
        <head>
            <script type="text/javascript" src=\""""
    HTML_FILE += VIS_location + """vis/dist/vis.js"></script>"""
    HTML_FILE += """
            <link href=""" + VIS_location + """vis/dist/vis.css rel="stylesheet" type="text/css" />
            <style type="text/css">
                #mynetwork
                {
                    width: 100%;
                    height: 90%;
                    border: 3px solid lightgray;
                }
            </style>
        </head>"""
    HTML_FILE += """
        <body>
            <div id = "mynetwork"></div>"""
    if(VIS_parameters[0]): # buttons
        HTML_FILE += """
            <p>These are different clustering methods (functions)</p>
            <input type = "button" onclick = "clusterByConnection()" value = "Cluster 'None' by connections">
            <br>
            <input type = "button" onclick = "clusterByHubsize()" value = "Cluster by hubsize">
            <br>
            <input type = "button" onclick = "clusterByColor()" value = "Cluster by color">"""
    if(VIS_parameters[10]):
        HTML_FILE += """
            <h2 style = "text-align:center;font-family:Helvetica;font-size:1.5em; margin-bottom: 2px;">"""
        HTML_FILE += str(init_strat_name) + str(edge_class).replace('_', ' ').upper() + " with " + str(strat_class).replace('_', ' ').upper() + "; Top " + str(top_CONST) + " Links </h2>"
    if(VIS_parameters[2]):
        HTML_FILE += """
            <div class='my-legend'>
                <div class='legend-scale'>
                    <ul class='legend-labels'>
                        """
        for num in range(len(color_strat_dict)):
            HTML_FILE += """<li><span style='background:""" + str(list(color_strat_dict.values())[num]) + """;'></span>""" + str(list(color_strat_dict.keys())[num]) + """</li>
                        """
        HTML_FILE += """
                </ul>
            </div>
        </div>
        <style type='text/css'>
          .my-legend {
            display: table;
            margin: 0 auto;
            margin-top: 0.5em;
            }
          .my-legend .legend-title {
            text-align: left;
            margin-bottom: 5px;
            font-weight: bold;
            font-size: 90%;
            }
          .my-legend .legend-scale ul {
            margin: 0;
            margin-bottom: 5px;
            padding: 0;
            float: left;
            list-style: none;
            }
          .my-legend .legend-scale ul li {
            font-size: 80%;
            list-style: none;
            line-height: 18px;
            margin-bottom: 5px;
            display: inline;
            float: left;
            margin-left: 1em;
            }
          .my-legend ul.legend-labels li span {
            display: block;
            float: left;
            height: 16px;
            width: 30px;
            margin-right: 5px;
            margin-left: 0;
            border: 1px solid #999;
            }
          .my-legend .legend-source {
            font-size: 70%;
            color: #999;
            clear: both;
            }
          .my-legend a {
            color: #777;
            }
        </style>
        """
    # HTML_FILE += """
    #         <div id = "mynetwork"></div>"""
    HTML_FILE += """
            <script type="text/javascript">"""

    # Loop for nodes, edges, options, functions (ONE+ INDENTS)
    HTML_FILE += """
                var nodes = new vis.DataSet([\n"""
    for node_num in range(len(dict_stratClass_edgeClass)):
        HTML_FILE += """                {"id": """ + str(node_num)
        if(VIS_parameters[4]): # node labels
            HTML_FILE += """, "label": """ + "\"" + str(d_keys_ind[node_num]) + "\""
        if(VIS_parameters[1]): # groups attribute
            HTML_FILE += """, "group": """ + "\"" + str(d_values_ind[node_num]) + "\""
        if(VIS_parameters[8]): # shapes
            HTML_FILE += """, "shape": """ + "\"" + str(shape_name) + "\""
        color_faded = [item + 'F2' for item in ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'][-15:]][node_num]
        HTML_FILE += """, color: {'border':'""" + color_faded + """', 'highlight':'""" + color_faded + "', " + "'hover':'" + color_faded + "'" + """}"""
        # if(VIS_parameters[9]): # node values/sizes
            # HTML_FILE += """, "value": """ + "\"" + someListUndefined + "\""
        if(node_num <= len(dict_stratClass_edgeClass) - 2):
            HTML_FILE += """},\n"""
        else:
            HTML_FILE += """}\n"""
    HTML_FILE += """            ]);"""
    HTML_FILE += """
                var edges = new vis.DataSet([\n"""
    c = 0
    for edge in edge_weights_3t:
        HTML_FILE += """                {"from": """ + str(d_keys_ind.index(edge[0]))
        HTML_FILE += """, "to": """ + str(d_keys_ind.index(edge[1]))
        HTML_FILE += """, "value": """ + str(edge[2])
        HTML_FILE += """, "color": {'inherit': 'both'}"""
        # if(edge_labels):
            # do something but there's no code here since what in the world would be the label (maybe the weight?)
        if(c <= len(edge_weights_3t) - 2):
            HTML_FILE += """},\n"""
        else:
            HTML_FILE += """}\n"""
        c += 1
    HTML_FILE += """            ]);"""
    HTML_FILE+="""
                var container = document.getElementById('mynetwork');
                var data = {
                    nodes: nodes,
                    edges: edges
                };"""
    HTML_FILE += """
                var options = {"""
    if(VIS_parameters[2]):
        HTML_FILE += """
                    groups: {\n"""
        for strat_num in range(len(list(color_strat_dict.keys()))):
            HTML_FILE += """\t\t\t\t\t\t\t\t\t\'""" + str(list(color_strat_dict.keys())[strat_num]) + "\': {color: {background: """ + "\"" + list(color_strat_dict.values())[strat_num] + """\"}, borderWidth: 1"""
            if(strat_num <= len(list(color_strat_dict.keys())) - 2):
                HTML_FILE += """},\n"""
            else:
                HTML_FILE += """}\n"""
        HTML_FILE += """                }"""
    if(VIS_parameters[3]): #physics option
        if(VIS_parameters[2]): HTML_FILE += "," #groups option
        HTML_FILE += """
                    "physics": {
                        "enabled": true,
                        "barnesHut": {
                            "theta": 0.5,
                            "gravitationalConstant": -2000,
                            "centralGravity": 0.3,
                            "springLength": 95,
                            "springConstant": 0.04,
                            "damping": 0.09,
                            "avoidOverlap": 0.1
                        },
                        "forceAtlas2Based": {
                            "theta": 0.5,
                            "gravitationalConstant": -50,
                            "centralGravity": 0.020,
                            "springConstant": 0.01,
                            "springLength": 100,
                            "damping": 0.4,
                            "avoidOverlap": 0.1
                        },
                        "repulsion": {
                            "centralGravity": 0.2,
                            "springLength": 200,
                            "springConstant": 0.05,
                            "nodeDistance": 100,
                            "damping": 0.09
                        },
                        "maxVelocity": 50,
                        "minVelocity": 0.1,
                        "solver": 'forceAtlas2Based',
                        "stabilization": {
                          "enabled": true,
                          "iterations": 1000,
                          "updateInterval": 100,
                          "onlyDynamicEdges": false,
                          "fit": true
                        },
                        "adaptiveTimestep": true
                    }"""
    if(VIS_parameters[5]): #manipulation option
        if(VIS_parameters[3]): HTML_FILE += "," #physics option
        HTML_FILE += """
                    "manipulation": {
                        "enabled": true,
                        "initiallyActive":false,
                        "addNode":true,
                        "addEdge":true,
                        "editNode":undefined,
                        "editEdge":true
                    }"""
    if(VIS_parameters[6]): #interaction option
        if(VIS_parameters[5]): HTML_FILE += "," #manipulation option
        HTML_FILE += """
                    "interaction": {
                        "dragNodes":true,
                        "dragView": true,
                        "hideEdgesOnDrag": false,
                        "hideEdgesOnZoom": true,
                        "hideNodesOnDrag": false,
                        "hover": true,
                        "hoverConnectedEdges": true,
                        "multiselect": true,
                        "navigationButtons": false,
                        "selectable": true,
                        "selectConnectedEdges": true,
                        "tooltipDelay": 0,
                        "zoomView": true
                    }"""
    if(VIS_parameters[7]): #layout option
        if(VIS_parameters[6]): HTML_FILE += "," #interaction option
        HTML_FILE += """
                    "layout": {
                        "improvedLayout": true,
                        "clusterThreshold":250
                    }"""
    HTML_FILE += """
                };"""
    HTML_FILE += """
                var network = new vis.Network(container, data, options)

                network.on("selectNode", function(params)
                {
                    if (params.nodes.length == 1)
                    {
                        if (network.isCluster(params.nodes[0]) == true)
                        {
                            network.openCluster(params.nodes[0]);
                        }
                    }
                });"""
    if(VIS_parameters[0]):
        HTML_FILE += """
                function clusterByConnection()
                {
                    network.setData(data);
                    network.clusterByConnection(1)
                }
                function clusterByHubsize()
                {
                    network.setData(data);
                    var clusterOptionsByData =
                    {
                        processProperties: function(clusterOptions, childNodes)
                        {
                            clusterOptions.label = "<" + childNodes.length + ">";
                            return clusterOptions;
                        },
                        clusterNodeProperties: {borderWidth:4, shape:'database', font:{size:29}}
                    };
                    network.clusterByHubsize(undefined, clusterOptionsByData);
                }
                function clusterByColor()
                {
                    network.setData(data);
                    var colors = ['blue','pink','green'];
                    var clusterOptionsByData;
                    for (var i = 0; i < colors.length; i++)
                    {
                        var color = colors[i];
                        clusterOptionsByData =
                        {
                            joinCondition: function (childOptions)
                            {
                            return childOptions.color.background == color;
                            },
                            processProperties: function (clusterOptions, childNodes, childEdges)
                            {
                                var totalMass = 0;
                                for (var i = 0; i < childNodes.length; i++)
                                {
                                    totalMass += childNodes[i].mass;
                                }
                                clusterOptions.mass = totalMass;
                                return clusterOptions;
                            },
                            clusterNodeProperties: {id: 'cluster:' + color, borderWidth: 3, shape: 'database', color:color, label:'color:' + color}
                        };
                        network.cluster(clusterOptionsByData);
                    }
                }"""
    HTML_FILE += """
            </script>
        </body>
    </html>"""

    return HTML_FILE

# Modify the 2D LIST for graph visualization
# >> Note: k > 64 makes colouring impossible
def list_clustering_graph_preproccessing(twod_list, K, top_CONST = 0):
    test = [nx.complete_graph(palette) for palette in twod_list]
    complete_EDGE_LIST = [tuple(sorted(tup)) for tup in list(chain.from_iterable([w.edges for w in test]))]

    # DETERMINING EDGE WEIGHTS
    edge_weights = collections.Counter(complete_EDGE_LIST)
    first_edge_value, second_edge_value = [i[0] for i in list(edge_weights)], [i[1] for i in list(edge_weights)]
    edge_weights_3t = list(zip(first_edge_value, second_edge_value, list(edge_weights.values()))) # based on edge weights (SORT, THEN TRUNCATE)
    if(top_CONST > 0):
        edge_weights_3t = sorted(edge_weights_3t, key = lambda tup: tup[2])[-top_CONST:]    # get the top edges
    else:
        edge_weights_3t = sorted(edge_weights_3t, key = lambda tup: tup[2])

    dict_stratClass_edgeClass = {}
    for i in range(15):
        dict_stratClass_edgeClass[i] = i

    d_keys_ind = list(dict_stratClass_edgeClass.keys())
    d_values_ind = list(dict_stratClass_edgeClass.values())

    # List of colors for each individual group based on strat class
    color_map = GLOBAL_DISTINCT_COLORS[-K:]
    color_strat_dict = dict(zip(list(set(d_values_ind)), color_map))

    return dict_stratClass_edgeClass, edge_weights_3t, d_keys_ind, d_values_ind, color_strat_dict

# Modify the dataframe for graph visualization
def graph_preproccessing(df, edge_class, strat_class, initial_stratifier_name, top_CONST = 0, return_level = 'minimal'):
    # Drop useless columns
    pd_Series_EDITED = df.groupby(by = [initial_stratifier_name], axis = 0)[edge_class]
    # Store all completegraphs in list, store all nodes in list, store all edges in list
    test = [nx.complete_graph(list(list(w)[1].values)) for w in pd_Series_EDITED]
    # Equate (re-order) A->B + B->A edges since Graph is Undirected
    complete_EDGE_LIST = [tuple(sorted(tup)) for tup in list(chain.from_iterable([w.edges for w in test]))]

    # Remove A->B + B->A edges since Graph is Undirected
    # if(graph_type == 'undirected'):
    #    complete_EDGE_LIST = list({tuple(frozenset(i)) for i in complete_EDGE_LIST})

    # DETERMINING EDGE WEIGHTS
    edge_weights = collections.Counter(complete_EDGE_LIST)
    first_edge_value, second_edge_value = [i[0] for i in list(edge_weights)], [i[1] for i in list(edge_weights)]
    edge_weights_3t = list(zip(first_edge_value, second_edge_value, list(edge_weights.values()))) # based on edge weights (SORT, THEN TRUNCATE)
    if(top_CONST > 0):
        edge_weights_3t = sorted(edge_weights_3t, key = lambda tup: tup[2])[-top_CONST:]    # get the top edges
    else:
        edge_weights_3t = sorted(edge_weights_3t, key = lambda tup: tup[2])
    if(len(edge_weights_3t) == 0):
        if(return_level == 'detailed'):
            return 0, 0, 0, 0, 0
        elif(return_level == 'minimal'):
            return 0, 0
        else:
            return -1
    # Edge-weight order and iterations are independent from node lists and node-strat_class dictionary
    nodes_from_edges_1 = [tup[0] for tup in edge_weights_3t]
    nodes_from_edges_2 = [tup[1] for tup in edge_weights_3t]
    nodes_from_edges = list(set(nodes_from_edges_1 + nodes_from_edges_2))

    # CONDENSE COLUMNS (keep edge class and strat class)
    # CONDENSE ROWS (keep ones in finalized node list)
    # Sort dictionary as same order as finalized node list
    # > Create the dictionary that defines the order for sorting
    # > Generate a rank column that will be used to sort the dataframe numerically
    # Remove escape characters in Dataframe columns
    # Convert from Dataframe to dictionary
    # Convert dictionary values from single-item list to string
    df__dict_stratClass_edgeClass = (df[[edge_class] + [strat_class]]).drop_duplicates(edge_class).reset_index().drop('index', 1)
    df__dict_stratClass_edgeClass = df__dict_stratClass_edgeClass[df__dict_stratClass_edgeClass[edge_class].isin(nodes_from_edges)].reset_index().drop('index', 1)
    sorterIndex = dict(zip(nodes_from_edges, range(len(nodes_from_edges))))
    df__dict_stratClass_edgeClass['edge_class temp rank'] = df__dict_stratClass_edgeClass[edge_class].map(sorterIndex)
    df__dict_stratClass_edgeClass.sort_values('edge_class temp rank', inplace = True)
    df__dict_stratClass_edgeClass.drop('edge_class temp rank', 1, inplace = True)
    df__dict_stratClass_edgeClass[edge_class] = df__dict_stratClass_edgeClass[edge_class].str.replace('"','\\"').str.replace("'","\\'")
    df__dict_stratClass_edgeClass[strat_class] = df__dict_stratClass_edgeClass[strat_class].str.replace('"','\\"').str.replace("'","\\'")
    dict_stratClass_edgeClass = df__dict_stratClass_edgeClass.set_index(edge_class).T.to_dict('list')
    dict_stratClass_edgeClass = {str(k):str(v[0]) for k, v in dict_stratClass_edgeClass.items()}

    edge_weights_3t = [(edge[0].replace('"','\\"').replace("'","\\'"), edge[1].replace('"','\\"').replace("'","\\'"), edge[2]) for edge in edge_weights_3t]

    d_keys_ind = list(dict_stratClass_edgeClass.keys())
    d_values_ind = list(dict_stratClass_edgeClass.values())

    # List of colors for each individual group based on strat class
    N = len(list(set(d_keys_ind)))
    # HSV_tuples = [[x*1.0/N, 0.5, 0.5] for x in range(N)]
    # RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    sns.set_palette('hls', n_colors = N)
    palette = sns.color_palette(n_colors = N)
    color_map = ['#%02x%02x%02x' % (int(proper_round(i * 255)), int(proper_round(j * 255)), int(proper_round(k * 255))) for i, j, k in palette]
    color_strat_dict = dict(zip(list(set(d_values_ind)), color_map))

    if(return_level == 'detailed'):
        return dict_stratClass_edgeClass, edge_weights_3t, d_keys_ind, d_values_ind, color_strat_dict
    elif(return_level == 'minimal'):
        return dict_stratClass_edgeClass, edge_weights_3t
    else:
        return -1
