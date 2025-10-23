// VLMS.js — VLMS 2.0 (practical, heavy, WebGL2 single-file engine)
// Exposes: window.VLMS = { Engine, Camera, Sphere, Light }
// Features: low-res path shader, motion vectors, temporal reprojection, separable bilateral denoise,
//           upsample + temporal blend, single-bounce reflections, dynamic quality switching.
// Requirements: WebGL2, EXT_color_buffer_float recommended.

(function(global){
  'use strict';

  // ---------- Utilities ----------
  function clamp(v,a,b){ return Math.max(a, Math.min(b, v)); }
  function isWebGL2(gl){ return !!gl && typeof WebGL2RenderingContext !== 'undefined' && gl instanceof WebGL2RenderingContext; }

  function compile(gl, type, src){
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)){
      const msg = gl.getShaderInfoLog(s);
      console.error('GLSL compile error:', msg);
      console.error(src);
      throw new Error(msg);
    }
    return s;
  }
  function link(gl, vsSrc, fsSrc){
    const vs = compile(gl, gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl, gl.FRAGMENT_SHADER, fsSrc);
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, 'a_pos');
    gl.linkProgram(p);
    if(!gl.getProgramParameter(p, gl.LINK_STATUS)){
      const msg = gl.getProgramInfoLog(p);
      console.error('Program link error:', msg);
      throw new Error(msg);
    }
    return p;
  }

  function makeFullQuad(gl){
    const v = new Float32Array([-1,-1, 1,-1, -1,1, -1,1, 1,-1, 1,1]);
    const b = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, b);
    gl.bufferData(gl.ARRAY_BUFFER, v, gl.STATIC_DRAW);
    return b;
  }

  function makeTexture(gl, w, h, internal = gl.RGBA32F, format = gl.RGBA, type = gl.FLOAT, linear=true){
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear?gl.LINEAR:gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear?gl.LINEAR:gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    return t;
  }

  function makeFBO(gl, texs){
    const f = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, f);
    const attachments = [];
    for(let i=0;i<texs.length;i++){
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + i, gl.TEXTURE_2D, texs[i], 0);
      attachments.push(gl.COLOR_ATTACHMENT0 + i);
    }
    gl.drawBuffers(attachments);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if(st !== gl.FRAMEBUFFER_COMPLETE) console.warn('FBO incomplete', st);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return f;
  }

  // ---------- GLSL sources ----------
  const VS_FULL = `#version 300 es
  layout(location=0) in vec2 a_pos;
  out vec2 v_uv;
  void main(){ v_uv = a_pos * 0.5 + 0.5; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

  // Low-res ray shader: outputs color (rgba), normal+depth (rgba), motion (rg)
  // Implements ray-marched spheres + plane, single-bounce reflection, and writes motion vectors
  const FS_RAY = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  layout(location=0) out vec4 outColor;
  layout(location=1) out vec4 outND;
  layout(location=2) out vec2 outMotion;

  uniform vec2 u_lowRes;
  uniform vec3 u_camPos;
  uniform vec3 u_camLook;
  uniform vec3 u_camUp;
  uniform vec3 u_prevCamPos;
  uniform vec3 u_prevCamLook;
  uniform vec3 u_prevCamUp;
  uniform float u_fov;
  uniform int u_frame;
  uniform int u_spp;
  uniform int u_bounces;
  uniform int u_sphereCount;
  uniform vec3 u_spherePos[16];
  uniform vec3 u_spherePrevPos[16];
  uniform float u_sphereR[16];
  uniform vec3 u_sphereColor[16];
  uniform float u_sphereRefl[16];

  // RNG basic
  uint wang_hash(uint seed){
    seed = (seed ^ 61u) ^ (seed >> 16);
    seed *= 9u;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15);
    return seed;
  }
  float rnd(inout uint s){ s = wang_hash(s); return float(s) / 4294967296.0; }

  struct Ray{ vec3 o; vec3 d; };
  struct Hit{ float t; vec3 n; int id; vec3 color; float refl; vec3 pos; };

  // sphere intersection
  bool intersectSphere(vec3 ro, vec3 rd, vec3 c, float r, out float t, out vec3 n){
    vec3 oc = ro - c;
    float b = dot(oc, rd);
    float c2 = dot(oc, oc) - r*r;
    float disc = b*b - c2;
    if(disc < 0.0) return false;
    float sq = sqrt(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    t = (t0 > 0.001) ? t0 : ((t1 > 0.001) ? t1 : -1.0);
    if(t < 0.0) return false;
    vec3 p = ro + rd*t;
    n = normalize(p - c);
    return true;
  }

  // plane intersection (y = -0.2 ground)
  bool intersectPlane(vec3 ro, vec3 rd, out float t, out vec3 n){
    vec3 pn = vec3(0.0, 1.0, 0.0);
    float denom = dot(rd, pn);
    if(abs(denom) < 1e-5) return false;
    t = (-0.2 - ro.y) / rd.y;
    if(t < 0.001) return false;
    n = pn;
    return true;
  }

  // scene intersect
  Hit sceneIntersect(vec3 ro, vec3 rd){
    Hit best; best.t = 1e20; best.id = -1; best.color = vec3(0.0); best.refl = 0.0; best.n = vec3(0.0); best.pos = vec3(0.0);
    for(int i=0;i<16;i++){
      if(i >= u_sphereCount) break;
      float t; vec3 n;
      if(intersectSphere(ro, rd, u_spherePos[i], u_sphereR[i], t, n)){
        if(t < best.t){
          best.t = t; best.n = n; best.id = i; best.color = u_sphereColor[i]; best.refl = u_sphereRefl[i];
          best.pos = ro + rd*t;
        }
      }
    }
    float tp; vec3 np;
    if(intersectPlane(ro, rd, tp, np)){
      if(tp < best.t){
        best.t = tp; best.n = np; best.id = 99; best.color = vec3(0.92); best.refl = 0.15; best.pos = ro + rd*tp;
      }
    }
    return best;
  }

  // camera basis
  void cameraBasis(vec3 look, vec3 up, out vec3 wz, out vec3 wx, out vec3 wy){
    wz = normalize(look);
    wx = normalize(cross(wz, up));
    wy = normalize(cross(wx, wz));
  }

  // project world point into camera (returns uv 0..1 or outside range)
  vec2 projectPointToUV(vec3 camPos, vec3 camLook, vec3 camUp, vec3 point, float fov, vec2 res){
    vec3 wz, wx, wy; cameraBasis(camLook, camUp, wz, wx, wy);
    vec3 toP = point - camPos;
    float z = dot(toP, wz);
    if(z <= 1e-5) return vec2(-10.0, -10.0);
    float aspect = res.x / res.y;
    float scale = tan(radians(fov * 0.5));
    float x = dot(toP, wx) / (z * scale * aspect);
    float y = dot(toP, wy) / (z * scale);
    vec2 ndc = vec2(x, -y);
    return ndc * 0.5 + 0.5;
  }

  // simple traced sample: primary + single reflection contribution
  vec3 traceSample(Ray r, inout uint seed, int maxBounces){
    vec3 accum = vec3(0.0);
    vec3 throughput = vec3(1.0);
    for(int bounce=0; bounce<=maxBounces; bounce++){
      Hit h = sceneIntersect(r.o, r.d);
      if(h.t > 9.9e18){
        accum += throughput * vec3(0.06, 0.08, 0.12); // environment
        break;
      }
      vec3 hitPos = h.pos; vec3 N = h.n;
      vec3 lightPos = vec3(0.0, 4.5, 2.0);
      vec3 toL = normalize(lightPos - hitPos);
      Hit sh = sceneIntersect(hitPos + N * 0.001, toL);
      float inLight = 1.0;
      if(sh.t < length(lightPos - hitPos)) inLight = 0.0;
      float lam = max(0.0, dot(N, toL));
      accum += throughput * h.color * lam * inLight * 1.0;
      // reflection probability based on material reflectivity
      float p = rnd(seed);
      if(h.refl > 0.0 && p < h.refl && bounce < maxBounces){
        vec3 refl = reflect(r.d, N);
        r.o = hitPos + N * 0.001;
        // jitter reflection slightly for roughness
        r.d = normalize(refl + vec3((rnd(seed)-0.5)*0.02, (rnd(seed)-0.5)*0.02, (rnd(seed)-0.5)*0.02));
        throughput *= 0.95;
        continue;
      } else {
        // diffuse bounce (simple cosine)
        float u = rnd(seed), v = rnd(seed);
        float phi = 2.0 * 3.141592653589793 * u;
        float cosT = sqrt(1.0 - v);
        float sinT = sqrt(max(0.0, 1.0 - cosT*cosT));
        vec3 tt = abs(N.z) < 0.999 ? normalize(cross(N, vec3(0,0,1))) : normalize(cross(N, vec3(0,1,0)));
        vec3 bb = normalize(cross(N, tt));
        vec3 newDir = normalize(N * cosT + (tt * cos(phi) + bb * sin(phi)) * sinT);
        r.o = hitPos + N * 0.001;
        r.d = newDir;
        throughput *= h.color * 0.9;
      }
    }
    return accum;
  }

  void main(){
    ivec2 pix = ivec2(gl_FragCoord.xy);
    uint seed = uint(u_frame) * 9781u + uint(pix.x)*1973u + uint(pix.y)*9277u;

    vec3 camWz, camWx, camWy; cameraBasis(u_camLook, u_camUp, camWz, camWx, camWy);
    float jx = rnd(seed)-0.5;
    float jy = rnd(seed)-0.5;
    float aspect = u_lowRes.x / u_lowRes.y;
    float scale = tan(radians(u_fov * 0.5));
    vec2 ndc = ((vec2(gl_FragCoord.xy) + vec2(jx,jy)) / u_lowRes) * 2.0 - 1.0;
    vec2 screen = vec2(ndc.x * aspect * scale, -ndc.y * scale);
    Ray cam; cam.o = u_camPos; cam.d = normalize(camWz + screen.x*camWx + screen.y*camWy);

    vec3 color = vec3(0.0);
    for(int s=0; s<u_spp; s++){
      color += traceSample(cam, seed, u_bounces);
    }
    color /= max(1, u_spp);

    // compute normal/depth/motion
    Hit last = sceneIntersect(cam.o, cam.d);
    vec3 normalOut = vec3(0.0);
    float depthOut = 1e5;
    vec2 motion = vec2(0.0);
    if(last.t < 1e19){
      normalOut = normalize(last.n);
      depthOut = last.t;
      // current uv of hit point in low-res
      vec2 curUV = projectPointToUV(u_camPos, u_camLook, u_camUp, last.pos, u_fov, u_lowRes);
      // compute previous world-space position for that object (if sphere, use prev pos; plane static)
      vec3 worldPrevPos = last.pos;
      if(last.id >= 0 && last.id < 16){
        // account for object velocity: reconstruct prev position by translating from current sphere to prev sphere pos offset
        vec3 curSpherePos = u_spherePos[last.id];
        vec3 prevSpherePos = u_spherePrevPos[last.id];
        vec3 offset = prevSpherePos - curSpherePos;
        worldPrevPos = last.pos + offset; // approximate
      }
      vec2 prevUV = projectPointToUV(u_prevCamPos, u_prevCamLook, u_prevCamUp, worldPrevPos, u_fov, u_lowRes);
      if(prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0){
        motion = prevUV - curUV;
      } else {
        motion = vec2(0.0);
      }
    }

    outColor = vec4(color, 1.0);
    outND = vec4(normalOut * 0.5 + 0.5, clamp(depthOut / 100.0, 0.0, 1.0));
    outMotion = motion;
  }`;

  // Temporal reprojection shader: reprojects prevAccum using motion (prevUV = curUV + motion), blends
  const FS_TEMPORAL = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 outAccum;
  uniform sampler2D u_curr; // current low-res color
  uniform sampler2D u_motion;
  uniform sampler2D u_prevAccum;
  uniform float u_feedback;
  void main(){
    vec2 uv = v_uv;
    vec3 curr = texture(u_curr, uv).rgb;
    vec2 motion = texture(u_motion, uv).xy;
    vec2 prevUV = uv + motion;
    vec3 prevCol = vec3(0.0);
    float valid = 0.0;
    if(all(greaterThanEqual(prevUV, vec2(0.0))) && all(lessThanEqual(prevUV, vec2(1.0)))){
      prevCol = texture(u_prevAccum, prevUV).rgb;
      valid = 1.0;
    }
    float diff = length(prevCol - curr);
    float adapt = smoothstep(0.0, 0.45, diff);
    float alpha = mix(u_feedback, 0.97, clamp(1.0 - adapt, 0.0, 1.0));
    vec3 outc = mix(curr, prevCol, alpha * valid);
    outAccum = vec4(outc, 1.0);
  }`;

  // Separable cross-bilateral denoiser
  const FS_DENOISE = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 frag;
  uniform sampler2D u_acc;
  uniform sampler2D u_nd;
  uniform vec2 u_texel;
  uniform int u_horizontal;
  void main(){
    vec2 uv = v_uv;
    vec3 center = texture(u_acc, uv).rgb;
    vec4 ndC = texture(u_nd, uv); vec3 nC = ndC.rgb; float dC = ndC.a;
    vec3 sum = vec3(0.0); float wsum = 0.0;
    for(int i=-3;i<=3;i++){
      vec2 off = (u_horizontal==1) ? vec2(float(i)*u_texel.x, 0.0) : vec2(0.0, float(i)*u_texel.y);
      vec3 c = texture(u_acc, uv + off).rgb;
      vec4 nd = texture(u_nd, uv + off); vec3 n = nd.rgb; float d = nd.a;
      float wc = exp(-dot(c-center,c-center)/(2.0*0.18*0.18));
      float wn = exp(-max(0.0,1.0-dot(nC,n))/(2.0*0.2*0.2));
      float wd = exp(-abs(d-dC)/(2.0*0.05*0.05));
      float w = wc * wn * wd;
      sum += c * w; wsum += w;
    }
    frag = vec4(sum / max(1e-6, wsum), 1.0);
  }`;

  // VLMS upscaler: bilateral gather from low-res + temporal blend
  const FS_VLMS = `#version 300 es
  precision highp float;
  in vec2 v_uv;
  out vec4 frag;
  uniform sampler2D u_low;
  uniform sampler2D u_nd;
  uniform sampler2D u_prevFull;
  uniform vec2 u_lowRes;
  uniform vec2 u_fullRes;
  uniform float u_feedback;
  uniform int u_enable;
  void main(){
    vec2 px = v_uv * u_fullRes;
    vec2 lowPx = floor(px * (u_lowRes / u_fullRes));
    vec2 lowUV = (lowPx + 0.5) / u_lowRes;
    vec3 base = texture(u_low, lowUV).rgb;
    vec4 nd = texture(u_nd, lowUV);
    vec3 n = nd.rgb * 2.0 - 1.0;
    if(u_enable == 0){ frag = vec4(base,1.0); return; }
    vec3 accum = vec3(0.0); float wsum = 0.0;
    for(int oy=-1; oy<=1; oy++) for(int ox=-1; ox<=1; ox++){
      vec2 off = vec2(float(ox), float(oy));
      vec2 sLow = ((floor(lowUV * u_lowRes) + off) + 0.5) / u_lowRes;
      vec3 c = texture(u_low, sLow).rgb;
      vec4 nd2 = texture(u_nd, sLow);
      vec3 n2 = nd2.rgb * 2.0 - 1.0;
      float sc = exp(-dot(c-base,c-base)/(2.0*0.2*0.2));
      float ss = exp(-dot(off,off)/(2.0*1.2*1.2));
      float sn = exp(-max(0.0,1.0-dot(n,n2))/0.2);
      float w = sc * ss * sn;
      accum += c * w; wsum += w;
    }
    vec3 up = accum / max(1e-6, wsum);
    vec3 hist = texture(u_prevFull, v_uv).rgb;
    float diff = length(hist - up);
    float adapt = smoothstep(0.0, 0.4, diff);
    float alpha = mix(u_feedback, 0.97, clamp(1.0 - adapt, 0.0, 1.0));
    vec3 outc = mix(up, hist, alpha);
    frag = vec4(outc,1.0);
  }`;

  // simple blit
  const FS_BLIT = `#version 300 es
  precision highp float;
  in vec2 v_uv; out vec4 frag;
  uniform sampler2D u_tex;
  void main(){ frag = texture(u_tex, v_uv); }`;

  // ---------- Scene classes ----------
  class Camera {
    constructor(opts = {}){
      this.pos = opts.pos || [0,1.2,-4];
      this.look = opts.look || [0,0.9,3.0];
      this.up = opts.up || [0,1,0];
      this.fov = opts.fov || 60;
      this.prevPos = [...this.pos];
      this.prevLook = [...this.look];
      this.prevUp = [...this.up];
    }
    copyPrev(){ this.prevPos = [...this.pos]; this.prevLook = [...this.look]; this.prevUp = [...this.up]; }
  }

  class Sphere {
    constructor(opts = {}){
      this.pos = opts.pos || [0,0.5,3];
      this.prevPos = [...this.pos];
      this.r = opts.radius || 0.5;
      this.color = opts.color || [1,0.2,0.2];
      this.refl = (typeof opts.refl === 'number') ? opts.refl : 0.2;
    }
    copyPrev(){ this.prevPos = [...this.pos]; }
  }

  class Light {
    constructor(opts = {}){
      this.pos = opts.pos || [2,4,2];
      this.color = opts.color || [1,1,1];
    }
  }

  // ---------- Engine ----------
  class Engine {
    constructor(canvasOrSelector, opts = {}){
      this.canvas = (typeof canvasOrSelector === 'string') ? document.querySelector(canvasOrSelector) : (canvasOrSelector || document.createElement('canvas'));
      if(!this.canvas) throw new Error('Canvas not found');
      this.gl = this.canvas.getContext('webgl2', { antialias:false });
      if(!isWebGL2(this.gl)) throw new Error('WebGL2 required');
      this.gl.getExtension('EXT_color_buffer_float') || console.warn('EXT_color_buffer_float missing — precision lower');

      this.quad = makeFullQuad(this.gl);

      // compile programs
      this.progRay = link(this.gl, VS_FULL, FS_RAY);
      this.progTemporal = link(this.gl, VS_FULL, FS_TEMPORAL);
      this.progDenoise = link(this.gl, VS_FULL, FS_DENOISE);
      this.progVLMS = link(this.gl, VS_FULL, FS_VLMS);
      this.progBlit = link(this.gl, VS_FULL, FS_BLIT);

      // scene
      this.camera = new Camera(opts.camera || {});
      this.spheres = [];
      this.lights = [];

      // settings
      this.scale = clamp(opts.scale || 0.5, 0.2, 1.0);
      this.spp = opts.spp || 1;
      this.bounces = opts.bounces || 1;
      this.enableVLMS = (opts.vlms !== undefined) ? !!opts.vlms : true;
      this.enableDenoise = (opts.denoise !== undefined) ? !!opts.denoise : true;
      this.feedback = opts.feedback || 0.85;

      this._frame = 0; this._accReset = true;
      this._accPing = 0; this._fullPing = 0;

      this._resizeTargets();
      window.addEventListener('resize', ()=>{ this._resizeTargets(); this.resetAccum(); });

      // default demo content (you can remove)
      if(opts.demo !== false){
        this.addSphere({ pos:[-1.1,0.9,1.5], radius:0.9, color:[0.85,0.2,0.2], refl:0.25 });
        this.addSphere({ pos:[0.6,0.6,0.8], radius:0.6, color:[0.15,0.45,0.9], refl:0.35 });
        this.addSphere({ pos:[1.6,0.9,2.2], radius:0.9, color:[0.95,0.95,1.0], refl:0.9 });
        this.addLight({ pos:[5,6,-2], color:[1,1,1] });
      }

      this._tick = this._tick.bind(this);
    }

    addSphere(opts){ const s = new Sphere(opts); this.spheres.push(s); return s; }
    addLight(opts){ const l = new Light(opts); this.lights.push(l); return l; }
    resetAccum(){ this._accReset = true; }
    setQuality(level){
      const map = { low:0.35, medium:0.6, high:0.9, ultra:1.0 };
      this.scale = map[level] || parseFloat(level) || this.scale;
      this.scale = clamp(this.scale, 0.2, 1.0);
      this._resizeTargets(); this.resetAccum();
    }
    setSPP(n){ this.spp = Math.max(1,n); this.resetAccum(); }
    setBounces(n){ this.bounces = Math.max(0,n); this.resetAccum(); }

    _resizeTargets(){
      const gl = this.gl;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const rc = this.canvas.getBoundingClientRect();
      this.fullW = Math.max(2, Math.floor(rc.width * dpr));
      this.fullH = Math.max(2, Math.floor(rc.height * dpr));
      this.lowW = Math.max(2, Math.floor(this.fullW * this.scale));
      this.lowH = Math.max(2, Math.floor(this.fullH * this.scale));

      // low-res: color, nd, motion
      this.texLowColor = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowND = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.texLowMotion = makeTexture(gl, this.lowW, this.lowH, gl.RG32F, gl.RG, gl.FLOAT);
      this.fboLow = makeFBO(gl, [this.texLowColor, this.texLowND, this.texLowMotion]);

      // accumulation ping-pong (low)
      this.texAccumA = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumA = makeFBO(gl, [this.texAccumA]);
      this.texAccumB = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboAccumB = makeFBO(gl, [this.texAccumB]);

      // denoised low
      this.texDenoised = makeTexture(gl, this.lowW, this.lowH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
      this.fboDenoised = makeFBO(gl, [this.texDenoised]);

      // final full-res ping-pong
      this.texFullA = makeTexture(gl, this.fullW, this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullA = makeFBO(gl, [this.texFullA]);
      this.texFullB = makeTexture(gl, this.fullW, this.fullH, gl.RGBA16F, gl.RGBA, gl.FLOAT);
      this.fboFullB = makeFBO(gl, [this.texFullB]);

      this._accPing = 0; this._fullPing = 0;
    }

    _bindQuad(prog){
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
      gl.useProgram(prog);
    }

    _passRay(){
      const gl = this.gl;
      this._bindQuad(this.progRay);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboLow);
      gl.viewport(0,0,this.lowW,this.lowH);

      const set3 = (name, v)=>{ const loc = gl.getUniformLocation(this.progRay, name); if(loc) gl.uniform3fv(loc, v); };
      gl.uniform2f(gl.getUniformLocation(this.progRay, 'u_lowRes'), this.lowW, this.lowH);
      set3('u_camPos', this.camera.pos); set3('u_camLook', this.camera.look); set3('u_camUp', this.camera.up);
      set3('u_prevCamPos', this.camera.prevPos); set3('u_prevCamLook', this.camera.prevLook); set3('u_prevCamUp', this.camera.prevUp);
      gl.uniform1f(gl.getUniformLocation(this.progRay, 'u_fov'), this.camera.fov);
      gl.uniform1i(gl.getUniformLocation(this.progRay, 'u_frame'), this._frame);
      gl.uniform1i(gl.getUniformLocation(this.progRay, 'u_spp'), this.spp);
      gl.uniform1i(gl.getUniformLocation(this.progRay, 'u_bounces'), this.bounces);
      gl.uniform1i(gl.getUniformLocation(this.progRay, 'u_sphereCount'), this.spheres.length);

      // push spheres (up to 16)
      for(let i=0;i<16;i++){
        const posLoc = gl.getUniformLocation(this.progRay, `u_spherePos[${i}]`);
        const prevLoc = gl.getUniformLocation(this.progRay, `u_spherePrevPos[${i}]`);
        const rLoc = gl.getUniformLocation(this.progRay, `u_sphereR[${i}]`);
        const cLoc = gl.getUniformLocation(this.progRay, `u_sphereColor[${i}]`);
        const reflLoc = gl.getUniformLocation(this.progRay, `u_sphereRefl[${i}]`);
        if(i < this.spheres.length){
          const s = this.spheres[i];
          gl.uniform3fv(posLoc, s.pos);
          gl.uniform3fv(prevLoc, s.prevPos);
          gl.uniform1f(rLoc, s.r);
          gl.uniform3fv(cLoc, s.color);
          gl.uniform1f(reflLoc, s.refl);
        } else {
          gl.uniform3fv(posLoc, [0,-9999,0]); gl.uniform3fv(prevLoc, [0,-9999,0]); gl.uniform1f(rLoc,0.0);
          gl.uniform3fv(cLoc, [0,0,0]); gl.uniform1f(reflLoc,0.0);
        }
      }

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passTemporal(){
      const gl = this.gl;
      const dstFBO = (this._accPing === 0) ? this.fboAccumA : this.fboAccumB;
      this._bindQuad(this.progTemporal);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.lowW,this.lowH);

      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texLowColor);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowMotion);
      const prevAccum = (this._accPing === 0) ? this.texAccumB : this.texAccumA;
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, prevAccum);

      gl.uniform1i(gl.getUniformLocation(this.progTemporal, 'u_curr'), 0);
      gl.uniform1i(gl.getUniformLocation(this.progTemporal, 'u_motion'), 1);
      gl.uniform1i(gl.getUniformLocation(this.progTemporal, 'u_prevAccum'), 2);
      gl.uniform1f(gl.getUniformLocation(this.progTemporal, 'u_feedback'), this.feedback);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passDenoise(){
      const gl = this.gl;
      if(!this.enableDenoise){
        // copy accum -> denoised
        this._bindQuad(this.progBlit);
        gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised);
        gl.viewport(0,0,this.lowW,this.lowH);
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
        gl.uniform1i(gl.getUniformLocation(this.progBlit, 'u_tex'), 0);
        gl.drawArrays(gl.TRIANGLES,0,6);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        return;
      }
      // horizontal
      this._bindQuad(this.progDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fboDenoised);
      gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.progDenoise, 'u_texel'), 1.0/this.lowW, 1.0/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_horizontal'), 1);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // vertical
      const dstFBO = (this._accPing === 0) ? this.fboAccumB : this.fboAccumA;
      this._bindQuad(this.progDenoise);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.lowW,this.lowH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texDenoised);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_acc'), 0);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_nd'), 1);
      gl.uniform2f(gl.getUniformLocation(this.progDenoise, 'u_texel'), 1.0/this.lowW, 1.0/this.lowH);
      gl.uniform1i(gl.getUniformLocation(this.progDenoise, 'u_horizontal'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _passVLMS(){
      const gl = this.gl;
      const dstFBO = (this._fullPing === 0) ? this.fboFullA : this.fboFullB;
      this._bindQuad(this.progVLMS);
      gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
      gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._accPing===0)?this.texAccumA:this.texAccumB);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.texLowND);
      gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, (this._fullPing===0)?this.texFullA:this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.progVLMS, 'u_low'), 0);
      gl.uniform1i(gl.getUniformLocation(this.progVLMS, 'u_nd'), 1);
      gl.uniform1i(gl.getUniformLocation(this.progVLMS, 'u_prevFull'), 2);
      gl.uniform2f(gl.getUniformLocation(this.progVLMS, 'u_lowRes'), this.lowW, this.lowH);
      gl.uniform2f(gl.getUniformLocation(this.progVLMS, 'u_fullRes'), this.fullW, this.fullH);
      gl.uniform1f(gl.getUniformLocation(this.progVLMS, 'u_feedback'), this.feedback);
      gl.uniform1i(gl.getUniformLocation(this.progVLMS, 'u_enable'), this.enableVLMS?1:0);
      gl.drawArrays(gl.TRIANGLES,0,6);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }

    _present(){
      const gl = this.gl;
      this._bindQuad(this.progBlit);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      gl.viewport(0,0,this.fullW,this.fullH);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, (this._fullPing===0)?this.texFullA:this.texFullB);
      gl.uniform1i(gl.getUniformLocation(this.progBlit, 'u_tex'), 0);
      gl.drawArrays(gl.TRIANGLES,0,6);
    }

    _tick(time){
      const rc = this.canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cw = Math.max(2, Math.floor(rc.width * dpr));
      const ch = Math.max(2, Math.floor(rc.height * dpr));
      if(this.canvas.width !== cw || this.canvas.height !== ch){
        this.canvas.width = cw; this.canvas.height = ch; this._resizeTargets(); this._accReset = true;
      }

      if(this._accReset){ this.camera.copyPrev(); for(const s of this.spheres) s.copyPrev(); }

      // example gentle animation (remove if you don't want)
      if(this.spheres.length>0){
        const t = time * 0.001;
        this.spheres[0].pos[0] = Math.sin(t*0.6) * 0.9;
      }

      // pipeline
      this._passRay();
      this._passTemporal();
      this._passDenoise();
      this._passVLMS();
      this._present();

      // swap
      this._accPing = 1 - this._accPing;
      this._fullPing = 1 - this._fullPing;
      this._accReset = false;

      for(const s of this.spheres) s.copyPrev();
      this.camera.copyPrev();
      this._frame++;
      requestAnimationFrame(this._tick);
    }

    start(){ requestAnimationFrame(this._tick); }
    stop(){ /* implement if needed */ }
  }

  // Export
  global.VLMS = { Engine, Camera, Sphere, Light };

})(window);