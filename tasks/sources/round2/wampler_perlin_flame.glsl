// shadertoy : https://www.shadertoy.com/view/XclBRM

/*
Copyright 2024 Taylor Wampler

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), 
to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// parameters
#define DRIFT 0.5
#define FLAME_WIDTH 0.15
#define FLAME_HEIGHT 0.8
#define FLAME_VERTICAL_SPREAD 0.5
#define FRACTAL_NOISE_INIT_FREQUENCY 2.0
#define FRACTAL_NOISE_OCTAVES 4
#define FRACTAL_NOISE_PERSISTENCE 0.7
#define INNER_COLOR vec3(0.9, 0.9, 0.7)
#define OUTER_COLOR vec3(0.1, 0.8, 0.3);

/*
REY, W. 1998. On generating random numbers, with help of y= [(a+x)sin(bx)] mod 1.
In 22nd European Meeting of Statisticians and the 7th Vilnius Conference on Probability
Theory and Mathematical Statistics. VSP, Zeist, The Netherlands.
*/
float rand(vec2 coord)
{
    float t = dot(vec2(19.244, 107.118), coord);
    
    return fract(sin(t) * 393919.062);
}

// A relatively smooth function for interpolation, first and second order derivatives are zero at t = 0 and t = 1.
/*
https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-5-implementing-improved-perlin-noise
*/
float quintic(float t) { return (6.0f * t * t - 15.0f * t + 10.0f) * t * t * t; }

/*
Ken Perlin. 1985. An image synthesizer.
In Proceedings of the 12th annual conference on Computer graphics and interactive techniques (SIGGRAPH '85). 
Association for Computing Machinery, New York, NY, USA, 287–296. https://doi.org/10.1145/325334.325247
*/
float perlinNoise(vec2 coord, float frequency)
{
  vec2 vec = vec2(coord.x * frequency, coord.y * frequency);
  float x1 = floor(vec.x);
  float y1 = floor(vec.y);
  float x2 = floor(vec.x + 1.0);
  float y2 = floor(vec.y + 1.0);
  float Lx = quintic(fract(vec.x));
  float Ly = quintic(fract(vec.y));

  return mix(
      mix(rand(vec2(x1, y1)), rand(vec2(x2, y1)), Lx),
      mix(rand(vec2(x1, y2)), rand(vec2(x2, y2)), Lx),
      Ly);
}

float fractalNoise(vec2 coord, float initFrequency, int steps, float persistence)
{
  float amplitude = 1.0;
  float frequency = initFrequency;
  float sum       = 0.0;
  float result    = 0.0;
  
  for (int i = 0; i < steps; i++)
  {
    frequency  *= 2.0;
    amplitude  *= persistence;
    sum        += amplitude;
    result     += amplitude * perlinNoise(coord, frequency);
  }
  result /= sum;
  
  
  return result;
}

// Use the Piriform Curve to measure 'distance'. 
// The curve is described by 0 = a^4 * x^2 + b^2 * y * (y - 2a)^3. 
// It has reflective symmetry about the y-axis and yMin = 0.
// Force a = height / 2 and b = 4 * width / (3 * sqrt(3)); with this tuning you get xMax = width / 2 and yMax = height.
/*
Weisstein, Eric W. "Piriform Curve." From MathWorld--A Wolfram Web Resource. https://mathworld.wolfram.com/PiriformCurve.html
*/
float piriform(vec2 coord, float width, float height)
{
    float a  = height / 2.0;
    float b  = 4.0 * width / (3.0 * sqrt(3.0));
    float yy = coord.y - 2.0 * a;
    
    return a * a * a * a * coord.x * coord.x + b * b * yy * yy * yy * coord.y; 
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv              = fragCoord.xy / iResolution.xy;
    float noise          = fractalNoise(uv - vec2(0.0, iTime * DRIFT), FRACTAL_NOISE_INIT_FREQUENCY, FRACTAL_NOISE_OCTAVES, FRACTAL_NOISE_PERSISTENCE);
    float verticalSpread = (2.0 * noise - 1.0) * 1.0 / (1.0 + exp(-uv.y)) * FLAME_VERTICAL_SPREAD; 
    vec2 coord           = vec2(abs(uv.x - 0.5), uv.y * (1.0 + verticalSpread)); 
    float inner          = -piriform(coord, 0.6 * FLAME_WIDTH, 0.5 * FLAME_HEIGHT);
    float outer          = -piriform(coord,       FLAME_WIDTH,       FLAME_HEIGHT);
    float innerShape     = step(0.0, inner);
    float outerShape     = step(0.0, outer);
    vec3 innerColor      = innerShape * INNER_COLOR;
    vec3 outerColor      = (1.0 - innerShape) * outerShape * OUTER_COLOR;
    
    fragColor = vec4(innerColor + outerColor, 1.0);
}