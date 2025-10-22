import random

import pandas
import numpy

import shapely
import matplotlib.pyplot as plt


random.seed( 32 )
shapes = []
positives = range(5)
scale = 25

cmap = ["red","green","blue","orange","yellow"]
cmap_counter = 0

immap = ["images/pexels-1561020_128.jpg", "images/pexels-17483848_128.jpg",
         "images/pexels-2157881_128.jpg", "images/pexels-23466423_128.jpg",
         "images/pexels-25626587_128.jpg", "images/pexels-25630351_128.jpg",
         "images/pexels-2693200_128.jpg" ]
immap_counter = 0

#Each iteration makes 10 polys, from 3 to 12
for batch in range(2):

    r = numpy.zeros( 12 )
    a = numpy.zeros( 12 )

    # three random lengths from 0,0 make a triangle
    r[0] = scale*random.random() + scale/2
    r[4] = scale*random.random() + scale/2
    r[8] = scale*random.random() + scale/2
    a = [i*numpy.pi/6 for i in range(12)]

    coords = [None]*12
    for i in [0,4,8]:
        coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))

    sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
    n = 3
    fname = f"{batch:02d}_{n:02d}.png"
    shapes.append( [fname,sh,n,n in positives] )

    fig = plt.figure( figsize=(0.7,0.7), dpi=128.0 )
    plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
    plt.axis( "off" )
    plt.savefig( f"{batch:02d}_{n:02d}_plain.png" )
    plt.close()

    img = plt.imread( immap[immap_counter] )
    immap_counter = (immap_counter+1)%len(immap)
    fig = plt.figure( figsize=(1,1), dpi=128.0 )
    plt.imshow( img, extent=[-32, 31, -32, 31])
    plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
    cmap_counter = (cmap_counter+1)%len(cmap)
    plt.axis( "off" )
    plt.savefig( fname )
    plt.close()

    # All indexes to the right of those already used
    # This will make 4,5,6
    for i in numpy.nonzero(r)[0] + 1:
        r[i] = scale*random.random() + scale/2
        coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))
        n += 1
        sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
        fname = f"{batch:02d}_{n:02d}.png"
        shapes.append( [fname,sh,n,n in positives] )

        fig = plt.figure( figsize=(1,1), dpi=128.0 )
        plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
        plt.axis( "off" )
        plt.savefig( f"{batch:02d}_{n:02d}_plain.png" )
        plt.close()

        img = plt.imread( immap[immap_counter] )
        immap_counter = (immap_counter+1)%len(immap)
        fig = plt.figure( figsize=(1,1), dpi=128.0 )
        plt.imshow( img, extent=[-32, 31, -32, 31])
        plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
        cmap_counter = (cmap_counter+1)%len(cmap)
        plt.axis( "off" )
        plt.savefig( fname )
        plt.close()
    

    # Repeat to make 7, 8, 9, 10, 11, 12
    for i in numpy.nonzero(r)[0] + 1:
        r[i] = scale*random.random() + scale/2
        coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))
        n += 1
        sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
        fname = f"{batch:02d}_{n:02d}.png"
        shapes.append( [fname,sh,n,n in positives] )

        fig = plt.figure( figsize=(1,1), dpi=128.0 )
        plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
        plt.axis( "off" )
        plt.savefig( f"{batch:02d}_{n:02d}_plain.png" )
        plt.close()

        img = plt.imread( immap[immap_counter] )
        immap_counter = (immap_counter+1)%len(immap)
        fig = plt.figure( figsize=(1,1), dpi=128.0 )
        plt.imshow( img, extent=[-32, 31, -32, 31])
        plt.plot( *sh.exterior.xy, color=cmap[cmap_counter] )
        cmap_counter = (cmap_counter+1)%len(cmap)
        plt.axis( "off" )
        plt.savefig( fname )
        plt.close()

labels = pandas.DataFrame( shapes, columns=["fname","shape","n","label"] )
labels.to_csv( "labels.csv" )

