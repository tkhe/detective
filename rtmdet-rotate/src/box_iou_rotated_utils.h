#ifndef BOX_IOU_ROTATE_H
#define BOX_IOU_ROTATE_H

#include <algorithm>
#include <cmath>

struct Point
{
    float x;
    float y;

    inline Point(const float &px = 0, const float &py = 0) : x(px), y(py) {}

    inline Point operator+(const Point &p) const
    {
        return Point(x + p.x, y + p.y);
    }

    inline Point operator+=(const Point &p)
    {
        x += p.x;
        y += p.y;
        return *this;
    }

    inline Point operator-(const Point &p) const
    {
        return Point(x - p.x, y - p.y);
    }

    inline Point operator*(const float coeff) const
    {
        return Point(x * coeff, y * coeff);
    }
};

inline float dot_2d(const Point &A, const Point &B)
{
    return A.x * B.x + A.y * B.y;
}

inline float cross_2d(const Point &A, const Point &B)
{
    return A.x * B.y - B.x * A.y;
}

int get_intersection_points(
    const Point (&pts1)[4],
    const Point (&pts2)[4],
    Point (&intersections)[24]
)
{
    // Line vector
    // A line from p1 to p2 is: p1 + (p2-p1)*t, t=[0,1]
    Point vec1[4];
    Point vec2[4];
    for (int i = 0; i < 4; i++)
    {
        vec1[i] = pts1[(i + 1) % 4] - pts1[i];
        vec2[i] = pts2[(i + 1) % 4] - pts2[i];
    }

    // When computing the intersection area, it doesn't hurt if we have
    // more (duplicated/approximate) intersections/vertices than needed,
    // while it can cause drastic difference if we miss an intersection/vertex.
    // Therefore, we add an epsilon to relax the comparisons between
    // the float point numbers that decide the intersection points.
    double EPS = 1e-5;

    // Line test - test all line combos for intersection
    int num = 0;  // number of intersections
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            // Solve for 2x2 Ax=b
            float det = cross_2d(vec2[j], vec1[i]);

            // This takes care of parallel lines
            if (fabs(det) <= 1e-14)
            {
                continue;
            }

            Point vec12 = pts2[j] - pts1[i];

            float t1 = cross_2d(vec2[j], vec12) / det;
            float t2 = cross_2d(vec1[i], vec12) / det;

            if ((t1 > -EPS) && (t1 < 1.0f + EPS) && (t2 > -EPS) && (t2 < 1.0f + EPS))
            {
                intersections[num++] = pts1[i] + vec1[i] * t1;
            }
        }
    }

    // Check for vertices of rect1 inside rect2
    {
        const Point &AB = vec2[0];
        const Point &DA = vec2[3];
        float ABdotAB = dot_2d(AB, AB);
        float ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++)
        {
            // assume ABCD is the rectangle, and P is the point to be judged
            // P is inside ABCD iff. P's projection on AB lies within AB
            // and P's projection on AD lies within AD
            const Point AP = pts1[i] - pts2[0];
            float APdotAB = dot_2d(AP, AB);
            float APdotAD = -dot_2d(AP, DA);
            if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) && (APdotAD < ADdotAD + EPS))
            {
                intersections[num++] = pts1[i];
            }
        }
    }

    // Reverse the check - check for vertices of rect2 inside rect1
    {
        const Point &AB = vec1[0];
        const Point &DA = vec1[3];
        auto ABdotAB = dot_2d(AB, AB);
        auto ADdotAD = dot_2d(DA, DA);
        for (int i = 0; i < 4; i++)
        {
            const Point AP = pts2[i] - pts1[0];
            float APdotAB = dot_2d(AP, AB);
            float APdotAD = -dot_2d(AP, DA);

            if ((APdotAB > -EPS) && (APdotAD > -EPS) && (APdotAB < ABdotAB + EPS) && (APdotAD < ADdotAD + EPS))
            {
                intersections[num++] = pts2[i];
            }
        }
    }

    return num;
}

int convex_hull_graham(
    const Point (&p)[24],
    const int &num_in,
    Point (&q)[24],
    bool shift_to_zero
)
{
    // Step 1:
    // Find point with minimum y
    // if more than 1 points have the same minimum y,
    // pick the one with the minimum x.
    int t = 0;
    for (int i = 1; i < num_in; i++)
    {
        if (p[i].y < p[t].y || (p[i].y == p[t].y && p[i].x < p[t].x))
        {
            t = i;
        }
    }
    const Point &start = p[t]; // starting point

    // Step 2:
    // Subtract starting point from every points (for sorting in the next step)
    for (int i = 0; i < num_in; i++)
    {
        q[i] = p[i] - start;
    }

    // Swap the starting point to position 0
    Point tmp = q[0];
    q[0] = q[t];
    q[t] = tmp;

    // Step 3:
    // Sort point 1 ~ num_in according to their relative cross-product values
    // (essentially sorting according to angles)
    // If the angles are the same, sort according to their distance to origin
    float dist[24];
    std::sort(
        q + 1,
        q + num_in,
        [](const Point &A, const Point &B) -> bool {
            float temp = cross_2d(A, B);
            if (fabs(temp) < 1e-6)
            {
                return dot_2d(A, A) < dot_2d(B, B);
            }
            else
            {
                return temp > 0;
            }
        }
    );
    // compute distance to origin after sort, since the points are now different.
    for (int i = 0; i < num_in; i++)
    {
        dist[i] = dot_2d(q[i], q[i]);
    }

    // Step 4:
    // Make sure there are at least 2 points (that don't overlap with each other)
    // in the stack
    int k;
    for (k = 1; k < num_in; k++)
    {
        if (dist[k] > 1e-8)
        {
            break;
        }
    }
    if (k == num_in)
    {
        // We reach the end, which means the convex hull is just one point
        q[0] = p[t];
        return 1;
    }
    q[1] = q[k];
    int m = 2;  // 2 points in the stack
    // Step 5:
    // Finally we can start the scanning process.
    // When a non-convex relationship between the 3 points is found
    // (either concave shape or duplicated points),
    // we pop the previous point from the stack
    // until the 3-point relationship is convex again, or
    // until the stack only contains two points
    for (int i = k + 1; i < num_in; i++)
    {
        while (m > 1)
        {
            const Point q1 = q[i] - q[m - 2];
            const Point q2 = q[m - 1] - q[m - 2];
            // cross_2d() uses FMA and therefore computes round(round(q1.x*q2.y) -
            // q2.x*q1.y) So it may not return 0 even when q1==q2. Therefore we
            // compare round(q1.x*q2.y) and round(q2.x*q1.y) directly. (round means
            // round to nearest floating point).
            if (q1.x * q2.y >= q2.x * q1.y)
            {
                m--;
            }
            else
            {
                break;
            }
        }
        // Using double also helps, but float can solve the issue for now.
        // while (m > 1 && cross_2d<T, double>(q[i] - q[m - 2], q[m - 1] - q[m - 2]) >= 0)
        // {
        //     m--;
        // }
        q[m++] = q[i];
    }

    // Step 6 (Optional):
    // In general sense we need the original coordinates, so we
    // need to shift the points back (reverting Step 2)
    // But if we're only interested in getting the area/perimeter of the shape
    // We can simply return.
    if (!shift_to_zero)
    {
        for (int i = 0; i < m; i++)
        {
            q[i] += start;
        }
    }
    return m;
}

float polygon_area(const Point (&q)[24], const int &m)
{
    if (m <= 2)
    {
        return 0;
    }

    float area = 0;
    for (int i = 1; i < m - 1; i++)
    {
        area += fabs(cross_2d(q[i] - q[0], q[i + 1] - q[0]));
    }
    return area / 2.0;
}

#endif
