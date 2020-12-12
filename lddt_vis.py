from __future__ import division

import sys, os
import pymol
from pymol import cmd
import gzip, bz2
import re
import string
import subprocess
from MoleculeUtils import colorCPK
import numpy as np
import colorsys

from pymol import cgo

# lddt visualizer by bcov in 2020

# loads commands:
# lddt
# lddt_interface
# lddt_path

# After you run lddt or lddt_interface. By pressing CTRL+C, you can easily toggle the estograms


my_lddt_path = "."

def set_lddt_path(path=None):
    global my_lddt_path

    if ( path is None ):
        print("Use lddt_path to set the path for the lddt npz files. Set it to the folder that contains them.")
        return

    my_lddt_path = path


def do_lddt():
    inner_lddt(interface=False)

def do_lddt_interface():
    inner_lddt(interface=True)

def inner_lddt(interface=False):

    # Save the camera
    save_view = cmd.get_view(output=1, quiet=1)

    pdbname = cmd.get_object_list()[0]

    file_path = os.path.join(my_lddt_path, pdbname + ".npz")

    if ( not os.path.exists(file_path) ):
        print("Could not find npz file: %s"%file_path)
        return

    dat = np.load(file_path)

    ### Stuff from Nao

    digitizations = [-20.0, -15.0, -10.0, -4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0, 10.0, 15.0, 20.0]
    masses = [digitizations[0]]+[(digitizations[i]+digitizations[i+1])/2 for i in range(len(digitizations)-1)]+[digitizations[-1]]

    def get_lddt(estogram, mask, center=7, weights=[1,1,1,1]):  
        # Remove diagonal from the mask.
        mask = np.multiply(mask, np.ones(mask.shape)-np.eye(mask.shape[0]))
        # Masking the estogram except for the last cahnnel
        masked = np.transpose(np.multiply(np.transpose(estogram, [2,0,1]), mask), [1,2,0])

        p0 = np.sum(masked[:,:,center], axis=-1)
        p1 = np.sum(masked[:,:,center-1]+masked[:,:,center+1], axis=-1)
        p2 = np.sum(masked[:,:,center-2]+masked[:,:,center+2], axis=-1)
        p3 = np.sum(masked[:,:,center-3]+masked[:,:,center+3], axis=-1)
        p4 = np.sum(mask, axis=-1)

        p4[p4==0] = 1
        # Only work on parts where interaction happen
        output = np.divide((weights[0]*p0 + weights[1]*(p0+p1) + weights[2]*(p0+p1+p2) + weights[3]*(p0+p1+p2+p3))/np.sum(weights), p4)
        return output

    #####

    # if interface, mask out the binder and the target to get individual lddts
    if ( interface ):

        blen = len(cmd.get_coords('name CA and chain A', 1))

        mask2 = np.zeros(dat['mask'].shape)
        mask2[:blen, blen:] = 1
        mask2[blen:, :blen] = 1

        my_lddt = get_lddt(dat['estogram'].transpose([1,2,0]), np.multiply(dat['mask'], mask2))

    else:

        my_lddt = get_lddt(dat['estogram'].transpose([1,2,0]), dat['mask'])


    print("========== Mean LDDT: %.2f"%np.mean(my_lddt))


    # colorspace for lddt visualization
    max_color = 121
    min_color = 0

    # interpolate on RGB so we don't see every color in between
    # set saturation to 0.5 so that we can distinguish from cylinders
    max_rgb = colorsys.hsv_to_rgb(max_color/360, 0.5, 1)
    min_rgb = colorsys.hsv_to_rgb(min_color/360, 0.5, 1)

    max_lddt = 1.0
    min_lddt = 0.5

    # color each residue the corresponding lddt color
    for seqpos in range(1, len(dat['mask'])):
        this_lddt = my_lddt[seqpos-1]
     
        r = np.interp(this_lddt, [min_lddt, max_lddt], [min_rgb[0], max_rgb[0]]) * 255
        g = np.interp(this_lddt, [min_lddt, max_lddt], [min_rgb[1], max_rgb[1]]) * 255
        b = np.interp(this_lddt, [min_lddt, max_lddt], [min_rgb[2], max_rgb[2]]) * 255

        color = "0x%02x%02x%02x"%(int(r), int(g), int(b))

        colorCPK("resi %i"%seqpos, color)

        # cmd.color(color, "resi %i"%seqpos,)


    # get 1 indexed ca positions
    cas = np.r_[ [[0, 0, 0]], cmd.get_coords('name CA', 1)]

    # max radius of cylinders (A)
    max_rad = 0.25

    # standard cylinder drawing function. color in HSV
    def get_cyl(start, end, rad_frac=1.0, color=[360, 0, 1]):

        rgb = colorsys.hsv_to_rgb(color[0]/360, color[1], color[2])

        radius = max_rad * rad_frac
        cyl = [
            # Tail of cylinder
            cgo.CYLINDER, start[0], start[1], start[2]
            , end[0], end[1], end[2]
            , radius, rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2]  # Radius and RGB for each cylinder tail
        ]

        return cyl


    def obj_exists(sele):
       return sele in cmd.get_names("objects")


    # clear out the old ones if we ran this twice
    for name in cmd.get_names("objects"):
        if (name.startswith("esto_res")):
            cmd.delete(name)


    estogram = np.transpose(dat['estogram'], [1, 2, 0])

    # parameters for cylinder colors
    # Since the lddt calculation maxes out at 4, we do so too
    # Also, fade near-0 to white so that we don't see sharp edges
    close_range_colors = [58, 0]
    far_range_colors = [167, 296]
    max_range = 4
    white_fade = 0.2

    # interpolate in hue space so that we do see all colors in-between
    def dist_to_color(dist):
        dist = np.clip(dist, -max_range, max_range)
        # dist = 2
        if ( dist < 0 ):
            bounds = close_range_colors
            dist = - dist
        else:
            bounds = far_range_colors

        hue = np.interp(dist, [0, max_range], bounds)
        sat = np.interp(dist, [0, white_fade], [0, 1])

        return [hue, sat, 1]

    # Mask is how likely the model things two residues are within 15A of each other
    # Don't draw cylinders for super distant residues
    min_mask = 0.1

    # actually draw the cylinders
    for seqpos in range(1, len(cas)):

        # pull out the estogram and mask for this position
        esto = estogram[seqpos-1] # shape (N x 15)
        mask = dat['mask'][seqpos-1] # shape (N)

        this_cgo = []

        for other in range(1, len(cas)):
            mask_edge = mask[other-1]
            if ( mask_edge < min_mask ):
                continue

            # estogram is a histogram of distances
            # this method comes directly from nao and takes the "center_of_mass" of that histogram
            esto_dist = np.sum(esto[other-1]*masses)

            color = dist_to_color(esto_dist)

            this_cgo += get_cyl(cas[seqpos], cas[other], rad_frac=mask_edge, color=color)

        if ( len(this_cgo) > 0):

            name = "esto_res%i"%seqpos
            cmd.load_cgo(this_cgo, name)
            cmd.disable(name)


    # disables all estograms and then enables the ones you've selected
    def enable_selected():
        selected = list(set([int(x.resi) for x in cmd.get_model("sele").atom]))

        for name in cmd.get_names("objects"):
            if (name.startswith("esto_res")):
                cmd.disable(name)

        for seqpos in selected:
            name = "esto_res%i"%seqpos
            cmd.enable(name)

        cmd.delete("sele")


    cmd.set_key( 'CTRL-C' , enable_selected )

    # restore the camera
    cmd.set_view(save_view)



cmd.extend( 'lddt', do_lddt )
cmd.extend( 'lddt_interface', do_lddt_interface )
cmd.extend( 'lddt_path', set_lddt_path )


