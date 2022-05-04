"""
Synthetic data pipeline
"""
from distutils.core import run_setup
from xml.sax.handler import DTDHandler
import bpy
import os
from bpy.props import (StringProperty,
                       CollectionProperty,
                       BoolProperty,
                       IntProperty,
                       FloatProperty,
                       FloatVectorProperty,
                       EnumProperty,
                       PointerProperty,
                       )
from bpy.types import (Panel,
                       Menu,
                       Operator,
                       PropertyGroup,
                       )
from bpy.utils import (previews)
from bpy import context
from bpy.types import Operator, Panel
from bpy.utils import register_class
from math import *
from mathutils import Euler
import itertools
import random

# --------------------------------------------------
# PARAMETERS
# --------------------------------------------------

rust_params = {

    # noise params, defines how much rust particles are made

    'noise_scale_val' : 7.1,
    'noise_detail_val' : 12.1,
    'noise_roughness_val' : .5,
    'noise_distortion_val' : 0,

    # vary how much rust shows up overall from here

    'cramp_1_val_1' : 0.750, #reverse element positions to get dramatic effect, vary from 0.3 - 0.7 each
    'cramp_1_val_2' : 0.598,

    'greater_than_val' : 0.8, #threshold
    'node_val_1' : 0.5, # vary from 0.1 to 50 for serious rust bumpage
    'node_val_2' : 0.02, # vary from 0.02 to 15 to increase major roughness 

    # color ramp 2 values define rust coloration
    'cramp_2_val_pos_1' : 0.03,
    'cramp_2_val_pos_2' : 0.105,
    'cramp_2_val_pos_3' : 0.504,
    'cramp_2_val_pos_4' : 1.0,
    'cramp2_color_1' : (0.00132, 0.00132, 0.00132, 1), #black
    'cramp2_color_2' : (0.3278, 0.13891, 0.0350, 1), #darker brown
    'cramp2_color_3' : (0.677, 0.4018, 0.197, 1), #lighter brown
    'cramp2_color_4' : (1, 1, 1, 1), # white
    'cramp_3_val_pos_1' : 0.464,
    'cramp_3_val_pos_2' : 0.442,

    'bump_val' : 0.1, # 0.1 to 10 for bump action

    'mix_to_bdsf_val' : 0.227, # vary from 0 to 0.3 

    'bsdf_sub_ior' : 1.4, # subsurface IOR
    'bsdf_sub_anio' : 0.0, # sub anisotropy
    'bsdf_metal' : 0.959, # metallic , vary from 0.7 - 0.99
    'bsdf_specular' : 0.191, # specular
    'bsdf_tint' : 0.441, # specular tint
    'bsdf_roughness' : 0.202, # roughness
    'bsdf_aniso' : 1.0, # anisotropic
    'bsdf_aniso_rot' : 0.0, # anisotropic rotation
    'bsdf_sheen' : 0.0, # sheen
    'bsdf_sheen_tint' : 0.5, # sheen tint
    'bsdf_clearcoat' : 0.0, # clearcoat
    'bsdf_clearcoat_roughness' : 0.03, # clearcoat roughness
    'bsdf_ior' : 1.45, # IOR
    'gt_base_color' : (0,0.5,0.00387,1), # green
    'gt_rust_color' : (0.5,0.001345,0,1), # red
    'cramp_4_val_pos_1' : 0.591,
    'cramp_4_val_pos_2' : 0.618
    }

tube_params = {
    'object_name' : 'thready',
    'VerticesPerLoop' : 512,
    'R' : 10.5, # Outer radius
    'r' : 10.25, # Inner radius
    'Loops' : 20, #150 looks cool too 
    'h1' : .2,
    'h2' : .1,
    'h3' : .2,
    'h4' : .2,
    'falloffRate' : 3
    }

# change it
file_path = r"C:/Users/Public/saved_images/"

# --------------------------------------------------
# THREADED TUBE OBJECT 
# --------------------------------------------------

def createMeshFromData(object_name, origin, verts, edges, faces):
    # Create mesh and object
    me = bpy.data.meshes.new(object_name)
    ob = bpy.data.objects.new(object_name, me)
    ob.location = origin
    ob.show_name = False
    # Link object to scene and make active
    bpy.context.collection.objects.link(ob)
    ob.select_set(True)
    # Create mesh from given verts, faces.
    me.from_pydata(verts, edges, faces)
    # Update mesh with new data
    me.update()

    return ob

def make_threaded_tube(
    object_name,
    VerticesPerLoop,
    R,
    r,
    Loops,
    h1,
    h2,
    h3,
    h4,
    falloffRate
    ):

    H = h1 + h2 + h3 + h4

    #build array of profile points
    ProfilePoints = []
    ProfilePoints.append( [r, 0, 0] )
    ProfilePoints.append( [R, 0, h1] )
    if h2 > 0:
        ProfilePoints.append( [R, 0, h1 + h2] )
    ProfilePoints.append( [r, 0, h1 + h2 + h3] )
    if h4 > 0:
        ProfilePoints.append( [r, 0, h1 + h2 + h3 + h4] )

    N = len(ProfilePoints)
    verts = [[0, 0, 0] for _ in range(N * (VerticesPerLoop + 1)  * Loops)]
    faces = [[0, 0, 0, 0] for _ in range(( N - 1) * VerticesPerLoop * Loops) ]

    # go around a cirle. for each point in ProfilePoints array, create a vertex
    angle = 0
    for i in range(VerticesPerLoop * Loops + 1):
        for j in range(N):
            angle = i * 2 * pi / VerticesPerLoop
            # falloff applies to outer rings only
            u = i / (VerticesPerLoop * Loops)
            radius = r + (R - r) * (1 - 6*(pow(2 * u - 1, falloffRate * 4)/2 - pow(2 * u - 1, falloffRate * 6)/3)) if ProfilePoints[j][0] == R else r

            x = radius * cos(angle)
            y = radius * sin(angle)
            z = ProfilePoints[j][2] + i / VerticesPerLoop * H

            verts[N*i + j][0] = x
            verts[N*i + j][1] = y
            verts[N*i + j][2] = z
    # now build face array
    for i in range(VerticesPerLoop * Loops):
        for j in range(N - 1):
            faces[(N - 1) * i + j][0] = N * i + j
            faces[(N - 1) * i + j][1] = N * i + 1 + j
            faces[(N - 1) * i + j][2] = N * (i + 1) + 1 + j
            faces[(N - 1) * i + j][3] =  N * (i + 1) + j

    return createMeshFromData(object_name, [0, 0, 0], verts, [], faces )

# --------------------------------------------------
# LIGHTING
# --------------------------------------------------

def making_lighting():

    # define light object
    light_datas = bpy.data.lights.new('light', type='POINT')
    light = bpy.data.objects.new('light', light_datas)
    # set light location
    light.location = (0,3.4,80)
    
    # link light object to collection
    bpy.context.collection.objects.link(light)
    # make light brighter
    bpy.context.collection.objects['light'].data.energy=5000000

    return light

# --------------------------------------------------
# CAMERA
# --------------------------------------------------

def add_camera():
    cam_data = bpy.data.cameras.new('camera')
    cam = bpy.data.objects.new('camera_s', cam_data)
    bpy.context.collection.objects.link(cam)
    #cam.location(50,-3,20)
    cam.location = (0,17,1.29)
    #cam.rotation_euler = Euler((-1.4,0,0), 'XYZ')

    # set camera to active

    bpy.context.scene.camera = cam

    # fit object to object

    for obj in context.scene.objects:
        obj.select_set(False)

    for obj in context.visible_objects:
        if not (obj.hide_get() or obj.hide_render):
            obj.select_set(True)

    bpy.ops.view3d.camera_to_view_selected()

    return cam

# --------------------------------------------------
# MATERIALS
# --------------------------------------------------

def assign_material(
    noise_scale_val,
    noise_detail_val,
    noise_roughness_val,
    noise_distortion_val,
    cramp_1_val_1,
    cramp_1_val_2,
    greater_than_val,
    node_val_1,
    node_val_2,
    cramp_2_val_pos_1,
    cramp_2_val_pos_2,
    cramp_2_val_pos_3,
    cramp_2_val_pos_4,
    cramp2_color_1,
    cramp2_color_2,
    cramp2_color_3,
    cramp2_color_4,
    cramp_3_val_pos_1,
    cramp_3_val_pos_2,
    bump_val,
    mix_to_bdsf_val,
    bsdf_sub_ior,
    bsdf_sub_anio,
    bsdf_metal,
    bsdf_specular,
    bsdf_tint,
    bsdf_roughness,
    bsdf_aniso,
    bsdf_aniso_rot,
    bsdf_sheen,
    bsdf_sheen_tint,
    bsdf_clearcoat,
    bsdf_clearcoat_roughness,
    bsdf_ior,
    gt_base_color,
    gt_rust_color, 
    cramp_4_val_pos_1, 
    cramp_4_val_pos_2

    ):

    # create new material

    material_metal_rust = bpy.data.materials.new(name="Mixed Metal and Rust")
    material_metal_rust.use_nodes = True

    # texture coordins
    texture_co_node = material_metal_rust.node_tree.nodes.new('ShaderNodeTexCoord')

    # noise texture

    noise_node = material_metal_rust.node_tree.nodes.new('ShaderNodeTexNoise')
    noise_node.inputs[1].default_value = noise_scale_val
    noise_node.inputs[2].default_value = noise_detail_val
    noise_node.inputs[3].default_value = noise_roughness_val
    noise_node.inputs[4].default_value = noise_distortion_val

    # color ramp 1, dictates amount of rust, if bumpy or indented

    colorRamp_node_1 = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    colorRamp_node_1.label = 'color ramp 1'
    colorRamp_node_1.color_ramp.elements[0].position = cramp_1_val_1 
    colorRamp_node_1.color_ramp.elements[1].position = cramp_1_val_2

    # greater than

    greater_than_node = material_metal_rust.node_tree.nodes.new("ShaderNodeMath")
    greater_than_node.operation = 'GREATER_THAN'
    greater_than_node.inputs[1].default_value = greater_than_val 

    # value
    
    value_node_1 = material_metal_rust.node_tree.nodes.new("ShaderNodeValue")
    value_node_1.label = 'value 1'
    value_node_1.outputs[0].default_value = node_val_1

    # value
    
    value_node_2 = material_metal_rust.node_tree.nodes.new("ShaderNodeValue")
    value_node_2.label = 'value 2'
    value_node_2.outputs[0].default_value = node_val_2

    # mix

    mixture_node_1 = material_metal_rust.node_tree.nodes.new("ShaderNodeMixRGB")
    mixture_node_1.label = 'mix 1'

    # color ramp 2, determines actual color of rust and base object

    colorRamp_node_2 = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    colorRamp_node_2.label = 'color ramp 2'
    colorRamp_node_2.color_ramp.elements[0].position = cramp_2_val_pos_1
    colorRamp_node_2.color_ramp.elements.new(cramp_2_val_pos_2)
    colorRamp_node_2.color_ramp.elements.new(cramp_2_val_pos_3)
    colorRamp_node_2.color_ramp.elements.new(cramp_2_val_pos_4)
    colorRamp_node_2.color_ramp.elements[0].color = cramp2_color_1
    colorRamp_node_2.color_ramp.elements[1].color = cramp2_color_2
    colorRamp_node_2.color_ramp.elements[2].color = cramp2_color_3
    colorRamp_node_2.color_ramp.elements[3].color = cramp2_color_4

    # color ramp 3

    colorRamp_node_3 = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    colorRamp_node_3.label = 'color ramp 3'
    colorRamp_node_3.color_ramp.elements[0].position = cramp_3_val_pos_1
    colorRamp_node_3.color_ramp.elements[1].position = cramp_3_val_pos_2

    # bump

    bump_node = material_metal_rust.node_tree.nodes.new("ShaderNodeBump")
    bump_node.inputs[1].default_value = bump_val

    # mix to bdsf

    mixture_node_2 = material_metal_rust.node_tree.nodes.new("ShaderNodeMixRGB")
    mixture_node_2.inputs[0].default_value = mix_to_bdsf_val
    mixture_node_2.label = 'mix 2'

    # bdsf

    bsdf_node = material_metal_rust.node_tree.nodes.get('Principled BSDF')

    bsdf_node.inputs[4].default_value = bsdf_sub_ior
    bsdf_node.inputs[5].default_value = bsdf_sub_anio
    bsdf_node.inputs[6].default_value = bsdf_metal
    bsdf_node.inputs[7].default_value = bsdf_specular
    bsdf_node.inputs[8].default_value = bsdf_tint
    bsdf_node.inputs[9].default_value = bsdf_roughness
    bsdf_node.inputs[10].default_value = bsdf_aniso
    bsdf_node.inputs[11].default_value = bsdf_aniso_rot
    bsdf_node.inputs[12].default_value = bsdf_sheen
    bsdf_node.inputs[13].default_value = bsdf_sheen_tint
    bsdf_node.inputs[14].default_value = bsdf_clearcoat
    bsdf_node.inputs[15].default_value = bsdf_clearcoat_roughness
    bsdf_node.inputs[16].default_value = bsdf_ior

    #bsdf_node.location(value=(364.502, -241.903, 0)) # set location rel to objects

    # invoke node switcher, to create ground truth images 

    switcher = material_metal_rust.node_tree.nodes.new(type="ShaderNodeGroup")
    switcher.node_tree = bpy.data.node_groups["mode_switcher"]

    # mix for ground truth

    mixture_node_3 = material_metal_rust.node_tree.nodes.new("ShaderNodeMixRGB")
    mixture_node_3.label = 'GT mix'
    mixture_node_3.inputs[1].default_value = gt_base_color
    mixture_node_3.inputs[2].default_value = gt_rust_color
    #mixture_node_3.inputs[3].default_value = (0.5,0.001345,0,1) # doesnt work

    ## @TODO - add two different colors for rust and object

    # color ramp for ground truth
    
    colorRamp_node_4 = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    colorRamp_node_4.label = 'GT ramp'
    colorRamp_node_4.color_ramp.interpolation = 'CONSTANT'
    colorRamp_node_4.color_ramp.elements[0].position = cramp_4_val_pos_1
    colorRamp_node_4.color_ramp.elements[1].position = cramp_4_val_pos_2

    # output material

    output_node = material_metal_rust.node_tree.get("Material Output")

    # output node sometimes is None, but it is there, look for it

    if output_node is None:

        # remove old material output node, its broken

        material_metal_rust.node_tree.nodes.remove(material_metal_rust.node_tree.nodes["Material Output"])

        # make a new one

        output_node = material_metal_rust.node_tree.nodes.new("ShaderNodeOutputMaterial")
        output_node.label = 'updated mat out'

    # make connections below

    material_metal_rust.node_tree.links.new(mixture_node_3.outputs[0], switcher.inputs[0]) 
    material_metal_rust.node_tree.links.new(colorRamp_node_4.outputs[0], mixture_node_1.inputs[0])
    material_metal_rust.node_tree.links.new(bsdf_node.outputs[0], switcher.inputs[0])
    material_metal_rust.node_tree.links.new(texture_co_node.outputs[3], noise_node.inputs[0])
    material_metal_rust.node_tree.links.new(noise_node.outputs[0], colorRamp_node_1.inputs[0])
    material_metal_rust.node_tree.links.new(noise_node.outputs[0], bump_node.inputs[2])
    material_metal_rust.node_tree.links.new(colorRamp_node_1.outputs[0], greater_than_node.inputs[0])
    material_metal_rust.node_tree.links.new(greater_than_node.outputs[0], mixture_node_1.inputs[0])
    material_metal_rust.node_tree.links.new(value_node_1.outputs[0], mixture_node_1.inputs[1])
    material_metal_rust.node_tree.links.new(value_node_2.outputs[0], mixture_node_1.inputs[2])
    material_metal_rust.node_tree.links.new(mixture_node_1.outputs[0], bump_node.inputs[0])
    material_metal_rust.node_tree.links.new(bump_node.outputs[0], bsdf_node.inputs[22]) # should be normal
    material_metal_rust.node_tree.links.new(colorRamp_node_1.outputs[0], colorRamp_node_2.inputs[0])
    material_metal_rust.node_tree.links.new(colorRamp_node_2.outputs[0], bsdf_node.inputs[0]) # should be base color
    material_metal_rust.node_tree.links.new(colorRamp_node_1.outputs[0], colorRamp_node_3.inputs[0])
    material_metal_rust.node_tree.links.new(colorRamp_node_3.outputs[0], mixture_node_2.inputs[1])
    material_metal_rust.node_tree.links.new(mixture_node_2.outputs[0], bsdf_node.inputs[9]) # should be roughness
    material_metal_rust.node_tree.links.remove(bsdf_node.outputs[0].links[0]) # remove one directly to output
    material_metal_rust.node_tree.links.new(switcher.outputs[0], output_node.inputs['Surface'])
    material_metal_rust.node_tree.links.new(mixture_node_3.outputs[0], switcher.inputs[1])
    material_metal_rust.node_tree.links.new(colorRamp_node_4.outputs[0], mixture_node_3.inputs[0])

    # connect noise to mode switcher

    material_metal_rust.node_tree.links.new(noise_node.outputs[0], colorRamp_node_4.inputs[0])
    material_metal_rust.node_tree.links.new(bsdf_node.outputs[0], switcher.inputs[0])

    bpy.context.view_layer.objects.active = bpy.data.objects["thready"]
    bpy.context.object.active_material = material_metal_rust

    # 4/26 - see if we can invoke 2nd class with this

    switcher2 = material_metal_rust.node_tree.nodes.new(type="ShaderNodeGroup")
    switcher2.node_tree = bpy.data.node_groups["mode_switcher"]

    # add other stuff from scratch

    # mapping

    mapping_node_s = material_metal_rust.node_tree.nodes.new('ShaderNodeMapping')
    mapping_node_s.label = "mapping node scratch"

    # wave texture 1

    wave_node_1_s = material_metal_rust.node_tree.nodes.new('ShaderNodeTexWave')
    wave_node_1_s.label = 'wave node 1 scratch'
    wave_node_1_s.inputs[1].default_value = 0.1 # scale
    wave_node_1_s.inputs[2].default_value = 78 # distortion 
    wave_node_1_s.inputs[3].default_value = 16 # 
    wave_node_1_s.inputs[5].default_value = 0.75 # detail roughness
    

    # noise texture

    noise_node_s = material_metal_rust.node_tree.nodes.new('ShaderNodeTexNoise')
    noise_node_s.label = "noise scratch"
    noise_node_s.inputs[2].default_value = 1 # scale
    noise_node_s.inputs[3].default_value = 10 # detail
    noise_node_s.inputs[4].default_value = 0.8 # roughness
    noise_node_s.inputs[5].default_value = 12 # distortion

    # wave texture 2 # making longer scratches downward

    wave_node_2_s = material_metal_rust.node_tree.nodes.new('ShaderNodeTexWave')
    wave_node_2_s.label = 'wave node 2 scratch'
    wave_node_2_s.inputs[1].default_value = 4 # scale
    wave_node_2_s.inputs[2].default_value = 4.5 # distortion
    wave_node_2_s.inputs[3].default_value = 2 # detail

    # color ramp 1

    color_ramp_node_1_s = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    color_ramp_node_1_s.label = 'color ramp 1 scratch'
    color_ramp_node_1_s.color_ramp.elements[0].color = (0.220025, 0.220025, 0.220025, 1)
    color_ramp_node_1_s.color_ramp.elements[1].color = (0.280313, 0.280313, 0.280313, 1)

    # color ramp 2

    color_ramp_node_2_s = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    color_ramp_node_2_s.label = 'color ramp 2 scratch'
    color_ramp_node_2_s.color_ramp.elements[0].position = 0
    color_ramp_node_2_s.color_ramp.elements[1].position = .591

    # color ramp 3

    color_ramp_node_3_s = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    color_ramp_node_3_s.label = 'color ramp 3 scratch'
    color_ramp_node_3_s.color_ramp.elements[0].position = .417
    color_ramp_node_3_s.color_ramp.elements[1].position = .410

    # separate xyz

    sep_xyz_node_s = material_metal_rust.node_tree.nodes.new("ShaderNodeSeparateXYZ")
    sep_xyz_node_s.label = 'scratch'

    # combine xyz

    combine_xyz_node_s = material_metal_rust.node_tree.nodes.new("ShaderNodeCombineXYZ")
    combine_xyz_node_s.label = 'scratch'

    #musgrave texture
    musgrave_node_s = material_metal_rust.node_tree.nodes.new("ShaderNodeTexMusgrave")
    musgrave_node_s.inputs[2].default_value = 3 # scale
    musgrave_node_s.inputs[3].default_value = 14 # detail
    musgrave_node_s.inputs[4].default_value = 0.56 # dimension
    musgrave_node_s.label = 'scratch'

    # bump 1
    bump_node_1_s = material_metal_rust.node_tree.nodes.new("ShaderNodeBump")
    bump_node_1_s.label = 'bump 1 scratch'
    bump_node_1_s.inputs[0].default_value = 0.07
    

    # bump 2
    bump_node_2_s = material_metal_rust.node_tree.nodes.new("ShaderNodeBump")
    bump_node_2_s.label = 'bump 2 scratch'
    bump_node_2_s.inputs[0].default_value = 0.3

    # bump 3
    bump_node_3_s = material_metal_rust.node_tree.nodes.new("ShaderNodeBump")
    bump_node_3_s.label = 'bump 3 scratch'

    # color ramp 4
    color_ramp_node_4_s = material_metal_rust.node_tree.nodes.new("ShaderNodeValToRGB")
    color_ramp_node_4_s.label = 'color ramp 4 scratch'
    color_ramp_node_4_s.color_ramp.elements[0].position = .7
    color_ramp_node_4_s.color_ramp.elements[1].position = .63 # use this to adjust amount of scratches in y-direction

    material_metal_rust.node_tree.links.new(texture_co_node.outputs[3], mapping_node_s.inputs[0]) # texture Obj [3] to Mapping Vector [0]
    material_metal_rust.node_tree.links.new(mapping_node_s.outputs[0], wave_node_1_s.inputs[0]) # mapping vector [0] to wave texture 1 [0] 
    material_metal_rust.node_tree.links.new(mapping_node_s.outputs[0], noise_node_s.inputs[0]) # mapping vector [0] to noise texture [0] 
    material_metal_rust.node_tree.links.new(mapping_node_s.outputs[0], wave_node_2_s.inputs[0]) # mapping vector [0] to wave texture 2 [0] 
    material_metal_rust.node_tree.links.new(mapping_node_s.outputs[0], sep_xyz_node_s.inputs[0]) # mapping vector [0] to separate xyz [0]
    material_metal_rust.node_tree.links.new(wave_node_1_s.outputs[0], color_ramp_node_1_s.inputs[0]) # wave texture 1 [0] to color ramp 1 [0]   
    material_metal_rust.node_tree.links.new(wave_node_1_s.outputs[0], color_ramp_node_2_s.inputs[0]) # wave texture 1 [0] to color ramp 2 [0]   
    material_metal_rust.node_tree.links.new(noise_node_s.outputs[0], color_ramp_node_3_s.inputs[0]) # noise texture  [0] to color ramp 3 [0]   
    material_metal_rust.node_tree.links.new(wave_node_2_s.outputs[0], combine_xyz_node_s.inputs[1]) # wave texture 2 [0] to combine xyz [1]
    material_metal_rust.node_tree.links.new(sep_xyz_node_s.outputs[0], combine_xyz_node_s.inputs[0]) # sepearate xyz [0] to combine xyz [0]
    material_metal_rust.node_tree.links.new(sep_xyz_node_s.outputs[2], combine_xyz_node_s.inputs[2]) # sepearate xyz [2] to combine xyz [2]
    material_metal_rust.node_tree.links.new(combine_xyz_node_s.outputs[0], musgrave_node_s.inputs[0]) # combine xyz [0] to musgrave texture [0]
    material_metal_rust.node_tree.links.new(musgrave_node_s.outputs[0], color_ramp_node_4_s.inputs[0]) # musgrave texture [0] to color ramp 4 [0]
    material_metal_rust.node_tree.links.new(color_ramp_node_4_s.outputs[0], bump_node_3_s.inputs[2]) # color ramp 4 [0] to bump 3 [2] height
    material_metal_rust.node_tree.links.new(color_ramp_node_1_s.outputs[0], bsdf_node.inputs[0]) # color ramp 1 [0] to bsdf [0] base color
    material_metal_rust.node_tree.links.new(color_ramp_node_2_s.outputs[0], bump_node_1_s.inputs[2]) # color ramp 2 [0] to bump 1 [2] height
    material_metal_rust.node_tree.links.new(bump_node_3_s.outputs[0], bsdf_node.inputs[22]) # bump 3 [0] to bsdf normal 

    material_metal_rust.node_tree.links.new(bump_node_2_s.outputs['Normal'], bump_node_3_s.inputs['Normal']) # bump 2 [0] to bump 3 [3] normal
    material_metal_rust.node_tree.links.new(bump_node_1_s.outputs['Normal'], bump_node_2_s.inputs['Normal']) # bump 1 [0] to bump 2 [3] normal
    material_metal_rust.node_tree.links.new(color_ramp_node_3_s.outputs[0], bump_node_2_s.inputs['Height']) # color ramp 3 [0] to bump 2 [2] height

    return material_metal_rust

def render_and_save(material_metal_rust, cnt):
    
    # apply material to object

    ob = bpy.context.active_object
    ob.data.materials[0] = material_metal_rust

    # real

    bpy.context.scene.render.filepath = file_path + '/real/'+str(cnt)+".png"
    bpy.ops.render.render(write_still=True,layer='real')
    
    toggle_mode(context) # assign materials again to see if GT changes. 

    ob = bpy.context.active_object
    ob.data.materials[0] = material_metal_rust

    # gt
    bpy.context.scene.render.filepath = file_path + '/gt/'+str(cnt)+".png"
    bpy.ops.render.render(write_still=True,layer='ground_truth')

    toggle_mode(context)

### ---- Mask and Ground Truth generating defs and classes --------------------


def create_mode_switcher_node_group():
    
    if 'mode_switcher' not in bpy.data.node_groups:

        print('no mode_switcher detected.., so we make one...')
        test_group = bpy.data.node_groups.new('mode_switcher', 'ShaderNodeTree')
        
        # create group inputs
        group_inputs = test_group.nodes.new('NodeGroupInput')
        group_inputs.location = (-350,0)
        test_group.inputs.new('NodeSocketShader','Real')
        test_group.inputs.new('NodeSocketColor','Ground Truth')
        #test_group.inputs.new('NodeSocketColor','Ground Truth class 2')

        # create group outputs
        group_outputs = test_group.nodes.new('NodeGroupOutput')
        group_outputs.location = (300,0)
        test_group.outputs.new('NodeSocketShader','Switch')

        # create three math nodes in a group
        node_mix = test_group.nodes.new('ShaderNodeMixShader')
        node_mix.location = (100,0)
        # adding mode driver
        # don't remove this, difficult to find
        modeDriver = bpy.data.node_groups['mode_switcher'].driver_add('nodes["Mix Shader"].inputs[0].default_value')
        modeDriver.driver.expression = 'mode'
        modeVariable = modeDriver.driver.variables.new()
        modeVariable.name = 'mode'
        modeVariable.type = 'SINGLE_PROP'
        modeVariable.targets[0].id_type = 'SCENE'
        modeVariable.targets[0].id = bpy.data.scenes['Scene']
        modeVariable.targets[0].data_path = 'uv_holographics.mode'

        # link inputs
        test_group.links.new(group_inputs.outputs['Real'], node_mix.inputs[1])
        test_group.links.new(group_inputs.outputs['Ground Truth'], node_mix.inputs[2])

        #link output
        test_group.links.new(node_mix.outputs[0], group_outputs.inputs['Switch'])

# ------------------------------------------------------------------------
#    PANEL IN OBJECT MODE
# ------------------------------------------------------------------------

import numpy as np
from random import uniform, randint

DEBUG = True
TEXTURE_RESOLUTION = 4 # this will be multiplied by 1024
MAIN_OBJECT_NAME = 'Target'
TARGET_COLLECTION = 'annotation'
       
def create_image(name, k=1):
    '''creates defect textures'''
    
    if name not in bpy.data.images:
        bpy.ops.image.new(name=name,
                            width=k*1024,
                            height=k*1024,
                            color=(0.0,0.0,0.0,0.0))
    else:
        log('-  create_image() : textures exists')
             
                            
def create_view_layers(context):
    '''todo: checks naming of view layers'''
    
    context.scene.view_layers[0].name = 'real'
    if 'Ground Truth' not in context.scene.view_layers:
        context.scene.view_layers.new(name='ground_truth')

def add_camera_focus(context, cameraName, target):
    camera = context.scene.objects[cameraName]
    
    if 'Track To' not in camera.constraints:
        tracker = camera.constraints.new(type='TRACK_TO')
        tracker.target = target
        tracker.track_axis = 'TRACK_NEGATIVE_Z'
        tracker.up_axis = 'UP_Y'
    else:
        log('-  add_camera_focus() : camera constraint already exists')
         
            
def toggle_mode(context):
    '''helper function for background switching'''
    
    scene = context.scene
    uvh = scene.uv_holographics
            
    if uvh.mode == 0:
        uvh.mode = 1
        scene.render.filter_size = 0
        scene.view_settings.view_transform = 'Standard'
    else:
        uvh.mode = 0
        scene.render.filter_size = 1.5
        scene.view_settings.view_transform = 'Filmic'
                
    # hack to update driver dependencies
    bpy.data.node_groups["mode_switcher"].animation_data.drivers[0].driver.expression = 'mode'
          

    
def run_variation(context):
    '''
    manipulates objects to create variations
    todo: scenarios
    todo: read from XML file
    '''
    
    # assume one camera
    camera = context.scene.objects['Camera']
    uvh = context.scene.uv_holographics
    
    # we assume a perimeter to sample our camera locations from
    r = uvh.camera_dist_mean + uniform(-uvh.camera_dist_var,uvh.camera_dist_var)
    theta = np.pi/2 + uniform(-np.pi/4,np.pi/8)
    phi = uniform(0,2*np.pi)
    
    # create parameter
    randX = r*np.sin(theta)*np.cos(phi)
    randY = r*np.sin(theta)*np.sin(phi)
    randZ = r*np.cos(theta)
    
    camera.location = (randX, randY, randZ)
    
                
# ------------------------------------------------------------------------
#    SCENE PROPERTIES, may need to delete 
# ------------------------------------------------------------------------

class MyProperties(PropertyGroup):
    separate_background: BoolProperty(
        name="Separate background class",
        description="Extra class for background",
        default = True
        )

    n_defects: IntProperty(
        name ="Defect classes",
        description="Number of defect classes",
        default = 1,
        min = 1,
        max = 10
        )
        
    n_samples: IntProperty(
        name ="Number of samples",
        description="Number of samples to generate",
        default = 1,
        min = 1,
        max = 500
        )
        
    target_object: PointerProperty(
        type =bpy.types.Object
        )
        
    target_collection: PointerProperty(
        type =bpy.types.Collection
        )

    output_dir: StringProperty(
        name = "Output folder",
        description="Choose a directory:",
        default="../output/",
        maxlen=1024,
        subtype='DIR_PATH'
        )
        
    mode: IntProperty(
        name ="Visualization mode",
        description="Realistic/Ground truth",
        default = 0,
        min = 0,
        max = 1
        )
        
    generate_real_only: BoolProperty(
        name="Generate real only",
        description="",
        default = False
        )
    
    # Camera
    
    min_camera_angle: FloatProperty(
        name = "Min camera angle",
        default = 0.,
        min = 0.,
        max = 1.,
        )
    max_camera_angle: FloatProperty(
        name = "Max camera angle",
        default = 1.,
        min = 0.,
        max = 1.,
        )
    camera_dist_mean: FloatProperty(
        name = "Camera dist mean",
        default = 5.,
        min = 0.,
        max = 10.,
        )
    camera_dist_var: FloatProperty(
        name = "Camera dist var",
        default = 1.,
        min = 0.,
        max = 4.,
        )

# ------------------------------------------------------------------------
#    Operators
# ------------------------------------------------------------------------
          
class WM_OT_GenerateComponents(Operator):
    '''Generate blank texture maps and view layers'''
    
    bl_label = "Generate components"
    bl_idname = "wm.gen_components"
    
    def execute(self, context):
        scene = context.scene
        uvh = scene.uv_holographics
        
        log('-  generating components')
        # create blank textures
        for i in range(uvh.n_defects):
            create_image(name=f"defect{i}",k=2)
            
        create_view_layers(context)
        #create_mode_switcher_node_group()
        add_camera_focus(context,'Camera',uvh.target_object) # updates viewport
        log('[[done]]')
        
        return {'FINISHED'}
                

class WM_OT_UpdateMaterials(Operator):
    '''Updates existing material nodes of objects in target_collection'''
    
    bl_label = "Update Materials"
    bl_idname = "wm.update_materials"
    
    def execute(self, context):
        
        for o in context.scene.uv_holographics.target_collection.objects:        
            for m in o.data.materials:
                insert_mode_switcher_node(context,m)
        
        return {'FINISHED'}
      
      
class WM_OT_ToggleMaterials(Operator):
    '''Toggle between realistic view and ground truth'''
    
    bl_label = "Toggle Real/GT"
    bl_idname = "wm.toggle_real_gt"
    
    def execute(self, context):
        toggle_mode(context)
        
        return {'FINISHED'}
    
class WM_OT_SampleVariation(Operator):
    '''Runs a sample variation'''
    
    bl_label = "Sample variation"
    bl_idname = "wm.sample_variation"
    
    def execute(self, context):
        run_variation(context)
        
        return {'FINISHED'}
            
class WM_OT_StartScenarios(Operator):
    '''Vary camera positions'''
    
    bl_label = "Generate"
    bl_idname = "wm.start_scenarios"
    
    def execute(self, context):
        scene = context.scene
        uvh = scene.uv_holographics
        
        # make sure to start in realistic mode
        if uvh.mode != 0:
            toggle_mode(context)
        
        for i in range(uvh.n_samples):
            run_variation(context)              
            render_layer(context, 'real', i+1)
            if not uvh.generate_real_only:
                toggle_mode(context)
                render_layer(context, 'ground_truth', i+1)
                toggle_mode(context)
                    
        return {'FINISHED'}

# ------------------------------------------------------------------------
#    Panel in Object Mode
# ------------------------------------------------------------------------

class OBJECT_PT_CustomPanel(Panel):
    bl_label = "Image Segmentation - Rust and Labels"
    bl_idname = "OBJECT_PT_custom_panel"
    bl_space_type = "VIEW_3D"   
    bl_region_type = "UI"
    bl_category = "Annotation"
    bl_context = "objectmode"
    
    @classmethod
    def poll(self,context):
        return context.object is not None
    
    def draw_header(self,context):
        global custom_icons
        self.layout.label(text="",icon_value=custom_icons["custom_icon"].icon_id)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        uvh = scene.uv_holographics
        layout.label(text='Setup')
        box = layout.box()
        box.prop(uvh, "separate_background")  
        box.prop(uvh, "n_defects")
        box.prop(uvh, "target_object")
        box.operator("wm.gen_components")
        box.prop(uvh, "target_collection")
        box.operator("wm.update_materials")
        layout.label(text='Camera parameters')
        box = layout.box()
        box.prop(uvh,"min_camera_angle", slider=True)
        box.prop(uvh,"max_camera_angle", slider=True)
        box.prop(uvh,"camera_dist_mean", slider=True)
        box.prop(uvh,"camera_dist_var", slider=True)
        layout.label(text='Operations')
        box = layout.box()
        box.operator("wm.toggle_real_gt")
        box.operator("wm.sample_variation")
        layout.label(text='Generation')
        box = layout.box()
        box.prop(uvh, "generate_real_only")
        box.prop(uvh, "n_samples")
        box.prop(uvh, "output_dir")
        box.operator("wm.start_scenarios")
        box.separator()

# ------------------------------------------------------------------------
#    Registration
# ------------------------------------------------------------------------

classes = (
    MyProperties,
    WM_OT_GenerateComponents,
    WM_OT_UpdateMaterials,
    WM_OT_ToggleMaterials,
    WM_OT_SampleVariation,
    WM_OT_StartScenarios,
    OBJECT_PT_CustomPanel
)

def register():

    from bpy.utils import register_class
    
    global custom_icons
        
    for cls in classes:
        register_class(cls)
        
    # https://blender.stackexchange.com/questions/32335/how-to-implement-custom-icons-for-my-script-addon
    custom_icons = bpy.utils.previews.new()
    script_path = bpy.context.space_data.text.filepath
    icons_dir = os.path.join(os.path.dirname(script_path), "icons")
    custom_icons.load("custom_icon", os.path.join(icons_dir, "logo.png"), 'IMAGE')
        
    bpy.types.Scene.uv_holographics = PointerProperty(type=MyProperties)

# --------------------------------------------------
# MAIN SEQUENCE
# --------------------------------------------------

def main():

    N = 5000 # number of image and mask pairs
    count = 0

    # make real and ground-truth folders

    if os.path.exists(file_path) != True:
        os.makedirs(file_path)

    if os.path.exists(file_path + 'real') != True:
        os.makedirs(file_path + 'real')

    if os.path.exists(file_path + 'gt') != True:
        os.makedirs(file_path + 'gt')

    for i in range(N):
    
    # for a, b, c, d, e in itertools.product(
    #     np.linspace(0.75, 0.3, 10), 
    #     np.linspace(0.598, 0.7, 10), 
    #     np.linspace(0.1, 10, 10), 
    #     np.linspace(0.02, 10, 10), 
    #     np.linspace(0.1, 10, 10)
    #     ):

        if count >= N:
            break

        else:
            
            # 1. change vals in dictionary

            print('working on: '+ str(count))

            # rust_params["cramp_1_val_1"] = a
            # rust_params["cramp_1_val_2"] = b
            # rust_params["node_val_1"] = c
            # rust_params["node_val_2"] = d
            # rust_params["bump_val"] = e

            rust_params["cramp_1_val_1"] = random.uniform(0,1)
            rust_params["cramp_1_val_2"] = random.uniform(0,1)
            rust_params["node_val_1"] = random.uniform(0,10)
            rust_params["node_val_2"] = random.uniform(0,10)
            rust_params["bump_val"] = random.uniform(0,10)

            rust_params["noise_scale_val"] = random.uniform(0,10)
            rust_params["noise_detail_val"] = random.uniform(0,20)
            rust_params["noise_roughness_val"] = random.uniform(0,1)

            # 2. make object

            ob = make_threaded_tube(**tube_params)

            # 3. rotate and move object

            ob.rotation_euler = (1.47159, 0, 0)    
            bpy.ops.transform.translate(value=(0,6.6952,0))

            # 4. Add shading/textures

            material_metal_rust = assign_material(**rust_params)

            # 5. make lighting

            light = making_lighting()

            # 6. make camera

            camera = add_camera()

            # 7. save output image

            render_and_save(material_metal_rust, count)

            # 8. delete object, light and camera

            ob.select_set(True)
            light.select_set(True)
            camera.select_set(True)
            bpy.ops.object.delete()
            
            count+=1


if __name__ == "__main__":

    register() # register classes
    create_mode_switcher_node_group() # create node switcher to gather ground truth data for masks
    main() # make data