# Basic Geomrtry

## Line
* Line: infinite line
* Line segement: two point form a line segment
* Ray: one point to another infinite end

## Polygon

* Convex
* Concave: at least one internal angle exceeding 180 degrees


## Polytope (多胞形) and Polyhedron (多面體)

## Equivalent vector

* same magnitude and the same direction

## Classify a point relative to a line segment

using cross product

* cross product > 0, left side
* cross product < 0, right side

## Dot product between a point and plane

Ax+By+Cz = D
(A, B, C) . (x, y, z) = D

* Dot product between a point on plane and normal vector on plane means the constant value D describe in plane

## Art gallery theorem

Aim to find number of guard in garden --> number of triangle in guard

* For simple polygons, number of triangles is N-2, where N is the number of edges in the polygon
* Guards = n / 3, where n is number of edges in the polygon

## Ear clipping triangulation

* Diagoanl of polygon (對角線)
* Ear of polygon: ear is convex vertex in polygon formed by v1, v2, v3 ; 

````
init ear status

while (util only three vertices are left) {
    pick a point and report a diagnal between neighbor vertices
    remove the picked vertex
    update the ear status of neighbor vertices
}
````

### Check diagonal

* line should not intersect with any other edges
* if start vertex of the line is a convex vertex, then two neighbors of the start vertex should lie on different sides to the line
* if the start vertex of the line is a reflex vertex, then it should not be an exterior line


