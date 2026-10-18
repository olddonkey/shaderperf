// CC0: Grid glow tracer
//  Shadertoy compatible
float fbm_noise(vec3 P) {
  float
    a=.6
  , d=0.
  , k=0.
  ;
  for(P*=.2;++k<4.;P=P.yzx) {
    // Used Dot noise by XorDev as an example.
    d+=a*(1.+dot(sin(P), cos(P*1.618).yzx));
    a*=.5;
    P.yz*=2.*mat2(.6,.8,-.8,.6);
  }

  return d;
}


void mainImage(out vec4 O, vec2 C) {
  float
    // Noise
  , d
    // Fade out factor
  , fo
    // Fade fog factor
  , ff
  ;
  vec2 
    r=iResolution.xy
  ;
  vec3
    // Accumulated output color
    o=vec3(0)
    // Current position
  , p
    // Ray direction
  , I=normalize(vec3(C-.5*r,r.y))
    // Ray origin
  , R=vec3(0,0,iTime)
    // Distance traveled per plane, have to adjust Z plane to make it smooth
  , Z=fract(-R)/I;
  ;
  // So strategy is to trace 2 things at the same time
  //  The Z planes
  //  The X planes
  //  By combining the two traces it will create a grid like pattern
  
  for(
      int i=0
      ,   j=0
    ; i<120
    ; ++i
    ) {
    // Flips between 2 and 0. 2 is the Z plane, 0 is the X plane
    // Sweet trick by FabriceNeyret2 to merge my original two loops into one
    j^=2;
    // Fade out factor
    fo=smoothstep(60.,40.,Z[j]);
    // Fade fog factor (makes lines less aliased)
    ff=smoothstep(10.,60.,Z[j]);
    // Step using current selected axis
    p=Z[j]*I;
    // Advance time with
    p+=R;
    // Compute fbm noise
    d=fbm_noise(p);
    // Makes noise more likes hills and canyons
    d+=.2*p.y;
    // Compute nice looking colors at the current position
    O=1.+sin(p.x*.1+p.y*.3+p.z*.2+vec4(2,7,0,2));
    // Tweak noise to avoid aliasing and suppress middle lines (which looks wierd)
    //  Random coding until it looks right
    d=abs(d)+5e-4/(I[j]*I[j])+1e-2*ff;
    // Brighter if noise is close to 0
    o+=O.w/d*O.xyz*fo;
    // Advance to next plane, due to j flipping twice we increment by .5 to get a full increment after two planes
    Z+=.5/abs(I);
  }
  // Reduce brightness and smooth with tanh
  O.xyz=tanh(o/5e2);
}
