import random

import pandas
import numpy

import shapely
import shapely.plotting
import matplotlib.pyplot as plt
import matplotlib #.colormaps as cmaps


random.seed( 16 )
shapes = []

max_r = 40
img_size = 128

cmap = matplotlib.colormaps.get_cmap("Greys")

immap = ["images/pexels-1561020_128.jpg", "images/pexels-17483848_128.jpg",
         "images/pexels-2157881_128.jpg", "images/pexels-23466423_128.jpg",
         "images/pexels-25626587_128.jpg", "images/pexels-25630351_128.jpg",
         "images/pexels-2693200_128.jpg" ]
immap_counter = 0


for n in range(25):

    r = (0.4+random.random()*0.6)*max_r
    c = ( r+random.random()*(img_size-2*r), r+random.random()*(img_size-2*r) )
    v1 = ( c[0],         c[1]+r )
    v2 = ( c[0]-r*0.866, c[1]-r/2 )
    v3 = ( c[0]+r*0.866, c[1]-r/2 )

    sh = shapely.affinity.rotate( shapely.geometry.Polygon( [v1,v2,v3] ),
                                  2*numpy.pi*random.random(),
                                  origin='centroid', use_radians=True )
    
    fname = f"{n:02d}.png"

    val = random.random()
    shapes.append( [fname,sh,val,val>0.8] )

    img = plt.imread( immap[immap_counter] )
    immap_counter = (immap_counter+1)%len(immap)
    fig,ax = plt.subplots( figsize=(1,1), dpi=img_size )
    ax.imshow( img, extent=[0, img_size-1, 0, img_size-1] )
    shapely.plotting.plot_polygon( sh, ax=ax, facecolor=cmap(val),
                                   linewidth=0.0, add_points=False )
    #cmap_counter = (cmap_counter+1)%len(cmap)
    ax.axis( "off" )
    fig.savefig( fname )
    plt.close()


labels = pandas.DataFrame( shapes, columns=["fname","shape","n","label"] )
labels.to_csv( "labels.csv" )

