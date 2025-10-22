import random

import pandas
import numpy

import shapely
import matplotlib.pyplot as plt


random.seed( 42 )
r = numpy.zeros( 12 )
a = numpy.zeros( 12 )
shapes = []

# three random lengths from 0,0 make a triangle
r[0] = random.random() + 0.5
r[4] = random.random() + 0.5
r[8] = random.random() + 0.5
a = [i*numpy.pi/6 for i in range(12)]

coords = [None]*12
for i in [0,4,8]:
    coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))

sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
n = 3
shapes.append( [sh,n] )

plt.plot( *sh.exterior.xy )
plt.axis( "off" )
plt.savefig( f"{n}.png" )
plt.close()

# All indexes to the right of those already used
# This will make 4,5,6
for i in numpy.nonzero(r)[0] + 1:
    # find the distance from 0,0 to sh at angle theta:
    # get a long line from 0,0 outwards
    l = shapely.geometry.LineString( [(0,0), (1*numpy.cos(a[i]), 1*numpy.sin(a[i]))] )
    cross = sh.intersection( l )
    # This is the part of l that is within the poly.
    # So the second coord is the intersection point, find the distance
    d = numpy.sqrt( cross.xy[0][1]**2 + cross.xy[1][1]**2 )
    # perturb to move the new vertex away from the existing edge
    r[i] = d + random.random() + 0.3
    coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))
    n += 1
    sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
    shapes.append( [sh,n] )

    plt.plot( *sh.exterior.xy )
    plt.axis( "off" )
    plt.savefig( f"{n}.png" )
    plt.close()

# Repeat to make 7, 8, 9, 10, 11, 12
for i in numpy.nonzero(r)[0] + 1:
    # find the distance from 0,0 to sh at angle theta:
    # get a long line from 0,0 outwards
    l = shapely.geometry.LineString( [(0,0), (1*numpy.cos(a[i]), 1*numpy.sin(a[i]))] )
    cross = sh.intersection( l )
    # This is the part of l that is within the poly.
    # So the second coord is the intersection point, find the distance
    d = numpy.sqrt( cross.xy[0][1]**2 + cross.xy[1][1]**2 )
    # perturb to move the new vertex away from the existing edge
    r[i] = d + random.random() + 0.3
    coords[i] = (r[i]*numpy.cos(a[i]), r[i]*numpy.sin(a[i]))
    n += 1
    sh = shapely.geometry.Polygon( [c for c in coords if c != None] )
    shapes.append( [sh,n] )

    plt.plot( *sh.exterior.xy )
    plt.axis( "off" )
    plt.savefig( f"{n}.png" )
    plt.close()

print( shapes )
