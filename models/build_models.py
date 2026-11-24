"""Primordia model builder.

Run with Blender (tested on 4.4):
    blender -b --python models/build_models.py

Builds every part and prop as a named root empty ("P_<name>") with sculpted
child meshes, saves models/primordia_models.blend, exports
models/primordia_parts.glb and renders models/preview.png.

Conventions (Blender axes): +Z is "outward" (the surface normal a part is
attached along, or "up" for props); +X is "forward" for feet and the bend
direction of hooks/curves. The game tints materials by name:
BODY, BODY_DARK, PATTERN, MEMBRANE, LEAF, EGG, FRUIT.
"""
import bpy, bmesh, math, random, os
from mathutils import Vector, noise

HERE = os.path.dirname(os.path.abspath(__file__))
random.seed(11)
bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COL = SC.collection

# ---------------------------------------------------------------- materials
MATS = {}
def hexc(h):
    h = h.lstrip('#'); return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))

def mat(name, hx, rough=.5, metal=0., emit=False, alpha=1., double=False):
    if name in MATS: return MATS[name]
    rgb = hexc(hx)
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes.get('Principled BSDF')
    b.inputs['Base Color'].default_value = (*rgb, 1.)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    if emit:
        b.inputs['Emission Color'].default_value = (*rgb, 1.)
        b.inputs['Emission Strength'].default_value = 1.
    if alpha < 1:
        b.inputs['Alpha'].default_value = alpha
        for attr, val in (('surface_render_method', 'BLENDED'), ('blend_method', 'BLEND')):
            try: setattr(m, attr, val)
            except Exception: pass
    m.use_backface_culling = not double
    m.diffuse_color = (*rgb, alpha)
    MATS[name] = m
    return m

BODY = lambda: mat('BODY', '#8fd0b8', .55, double=True)
BODY_DARK = lambda: mat('BODY_DARK', '#4f7f70', .6)
PATTERN = lambda: mat('PATTERN', '#f2e6a0', .5, double=True)
MEMBRANE = lambda: mat('MEMBRANE', '#f2e6a0', .4, alpha=.78, double=True)
IVORY = lambda: mat('IVORY', '#efe6cc', .35)
EYE_W = lambda: mat('EYE_WHITE', '#f7f4ec', .15)
PUPIL = lambda: mat('PUPIL', '#0d0d0d', .1)
BEAK = lambda: mat('BEAK', '#f0a63a', .35)
MOUTH = lambda: mat('MOUTH', '#55131a', .6)
TONGUE = lambda: mat('TONGUE', '#e06a7a', .4)
SAC = lambda: mat('SAC', '#ff8cbf', .15, alpha=.85)
BARK = lambda: mat('BARK', '#6b4a2a', .9)
LEAF = lambda: mat('LEAF', '#4f8a35', .8, double=True)
ROCK = lambda: mat('ROCK', '#8a8a82', .95)
TWIG = lambda: mat('TWIG', '#7a5532', .9)
NEST_SOFT = lambda: mat('NEST_SOFT', '#b8a07a', 1.)
EGG = lambda: mat('EGG', '#e8e0c8', .4)
FRUIT = lambda: mat('FRUIT', '#e04a3a', .3)
MEAT = lambda: mat('MEAT', '#b8423c', .45)
FAT = lambda: mat('FAT', '#f0d6c8', .5)
ALGAE = lambda: mat('ALGAE', '#6fd65a', .6)
CRYSTAL = lambda: mat('CRYSTAL', '#7ff0e0', .1, emit=True)
FOSSIL = lambda: mat('FOSSIL', '#c9b99a', .8)
STONE = lambda: mat('STONE', '#7d7f78', .95)
ORGANELLE = lambda: mat('ORGANELLE', '#e89a5a', .4)
FROND = lambda: mat('FROND', '#9be07a', .5)

# ---------------------------------------------------------------- helpers
def mk(name, bm, m, parent, smooth=True):
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    for p in me.polygons: p.use_smooth = smooth
    ob = bpy.data.objects.new(name, me); COL.objects.link(ob)
    me.materials.append(m); ob.parent = parent
    return ob

_slot = [0]
def root(name, size=1.):
    """Create a root empty, laid out on a grid in the .blend for browsing."""
    i = _slot[0]; _slot[0] += 1
    e = bpy.data.objects.new('P_' + name, None); COL.objects.link(e)
    e.location = ((i % 8) * 4.5, -(i // 8) * 6.5, 0); e.empty_display_size = .3
    return e

def spline(ctrl, n):
    """Catmull-Rom through control points, n samples."""
    pts = [Vector(c) for c in ctrl]
    ext = [pts[0] * 2 - pts[1]] + pts + [pts[-1] * 2 - pts[-2]]
    out = []
    segs = len(pts) - 1
    for s in range(n):
        u = s / (n - 1) * segs; i = min(int(u), segs - 1); t = u - i
        a, b, c, d = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        t2, t3 = t * t, t * t * t
        out.append(.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3))
    return out

def loft(path, radii, segs=16, prof=None, cap0=True, cap1=True):
    """Sweep a ring along a path with parallel-transport frames.
    radii: float or (rx, ry) per point. prof(t, angle) -> radius multiplier."""
    bm = bmesh.new(); N = len(path); T = []
    for i in range(N):
        t = path[min(i + 1, N - 1)] - path[max(i - 1, 0)]; t.normalize(); T.append(t)
    ref = Vector((0, 0, 1)) if abs(T[0].z) < .9 else Vector((1, 0, 0))
    x = T[0].cross(ref); x.normalize(); rings = []
    for i in range(N):
        t = T[i]; x = x - t * x.dot(t)
        if x.length < 1e-6: x = t.orthogonal()
        x.normalize(); y = t.cross(x)
        r = radii[i]; rx, ry = r if isinstance(r, tuple) else (r, r)
        ring = []
        for k in range(segs):
            a = k / segs * math.tau
            m = prof(i / (N - 1), a) if prof else 1.
            ring.append(bm.verts.new(path[i] + x * math.cos(a) * rx * m + y * math.sin(a) * ry * m))
        rings.append(ring)
    for i in range(N - 1):
        for k in range(segs):
            bm.faces.new((rings[i][k], rings[i][(k + 1) % segs], rings[i + 1][(k + 1) % segs], rings[i + 1][k]))
    if cap0:
        c = bm.verts.new(path[0] - T[0] * 1e-4)
        for k in range(segs): bm.faces.new((rings[0][(k + 1) % segs], rings[0][k], c))
    if cap1:
        c = bm.verts.new(path[-1] + T[-1] * 1e-4)
        for k in range(segs): bm.faces.new((rings[-1][k], rings[-1][(k + 1) % segs], c))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm

def lathe(profile, segs=20, n=None, prof=None):
    """profile: [(r, z)] bottom to top, smoothed with a spline."""
    pts = spline([(r, 0, z) for r, z in profile], n or len(profile) * 4)
    return loft([Vector((0, 0, p.z)) for p in pts], [max(p.x, 1e-4) for p in pts], segs, prof)

def taper(path, r0, r1=.003, power=1., segs=12, prof=None):
    N = len(path)
    return loft(path, [r1 + (r0 - r1) * (1 - i / (N - 1)) ** power for i in range(N)], segs, prof)

def blob(r=1., sub=3, scale=(1, 1, 1), loc=(0, 0, 0), amp=0., freq=1., seed=0, oct=3):
    bm = bmesh.new(); bmesh.ops.create_icosphere(bm, subdivisions=sub, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    if amp: sculpt(bm, amp, freq, seed, oct)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts)
    return bm

def sphere(r, loc=(0, 0, 0), scale=(1, 1, 1), u=20, v=14):
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts)
    return bm

def sculpt(bm, amp, freq, seed=0, oct=3):
    """Sculpt-style organic displacement along vertex normals."""
    bm.normal_update(); off = Vector((seed * 13.1, seed * 7.7, seed * 3.3))
    for v in bm.verts:
        v.co += v.normal * amp * noise.fractal(v.co * freq + off, 1., 2., oct)

def xform(bm, loc=(0, 0, 0), rot=None):
    if rot: bmesh.ops.rotate(bm, cent=Vector((0, 0, 0)), matrix=rot, verts=bm.verts)
    bmesh.ops.translate(bm, vec=Vector(loc), verts=bm.verts)
    return bm

def torus(R, r, maj=24, mnr=10, sz=1.):
    bm = bmesh.new(); vs = []
    for i in range(maj):
        a = i / maj * math.tau; row = []
        for j in range(mnr):
            b = j / mnr * math.tau
            row.append(bm.verts.new(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b) * sz)))
        vs.append(row)
    for i in range(maj):
        for j in range(mnr):
            bm.faces.new((vs[i][j], vs[(i + 1) % maj][j], vs[(i + 1) % maj][(j + 1) % mnr], vs[i][(j + 1) % mnr]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm

def surface(fn, nu, nv):
    """Grid surface from fn(u, v) -> (x, y, z)."""
    bm = bmesh.new()
    g = [[bm.verts.new(fn(i / nu, j / nv)) for j in range(nv + 1)] for i in range(nu + 1)]
    for i in range(nu):
        for j in range(nv):
            bm.faces.new((g[i][j], g[i + 1][j], g[i + 1][j + 1], g[i][j + 1]))
    return bm

def rotm(axis, ang):
    from mathutils import Matrix
    return Matrix.Rotation(ang, 3, axis)

# ---------------------------------------------------------------- parts
def eye_into(e, s=1., off=Vector((0, 0, 0)), lid=True, pupil=(1, 1, .55), iris=None):
    mk(e.name + '_white', xform(sphere(.26 * s), off + Vector((0, 0, .05 * s))), EYE_W(), e)
    mk(e.name + '_iris', xform(sphere(.14 * s, scale=(1, 1, .35)), off + Vector((0, 0, .29 * s))), (iris or PATTERN)(), e)
    mk(e.name + '_pupil', xform(sphere(.075 * s, scale=pupil), off + Vector((0, 0, .325 * s))), PUPIL(), e)
    if lid:
        bm = sphere(.295 * s, u=24, v=16)
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.y < -.02 * s or v.co.z > .17 * s], context='VERTS')
        mk(e.name + '_lid', xform(bm, off + Vector((0, 0, .05 * s))), BODY(), e)

def m_eye():
    e = root('eye'); eye_into(e)

def m_stalkeye():
    e = root('stalkeye')
    path = spline([(0, 0, 0), (.03, 0, .22), (.09, 0, .44), (.1, 0, .6)], 14)
    N = len(path)
    mk('stalk', loft(path, [.12 - .07 * min(1, i / (N * .3)) - .01 * i / N for i in range(N)], 12), BODY(), e)
    eye_into(e, .9, Vector((.1, 0, .58)))

def m_beak():
    e = root('beak')
    up = spline([(0, 0, 0), (.03, 0, .25), (.1, 0, .48), (.22, 0, .6)], 16); N = len(up)
    mk('upper', loft(up, [(.25 * (1 - i / N) ** .8 + .004, .16 * (1 - i / N) ** .9 + .003) for i in range(N)], 16), BEAK(), e)
    lo = spline([(.08, 0, 0), (.11, 0, .2), (.15, 0, .38)], 12); N = len(lo)
    mk('lower', loft(lo, [(.2 * (1 - i / N) ** .8 + .004, .1 * (1 - i / N) + .003) for i in range(N)], 14), BEAK(), e)
    mk('cere', torus(.23, .05, 24, 8, .7), BODY(), e)

def mouth_into(e, s=1., big=False):
    lips = torus(.28 * s, .075 * s, 28, 10, .7); sculpt(lips, .01 * s, 6, 3)
    mk(e.name + '_lips', lips, BODY(), e)
    mk(e.name + '_inside', sphere(.26 * s, scale=(1, 1, .35)), MOUTH(), e)
    fangs = [60, 120, 240, 300]
    for k, deg in enumerate(fangs + ([0, 180] if big else [])):
        a = math.radians(deg); p0 = Vector((math.cos(a) * .22 * s, math.sin(a) * .22 * s, .03 * s))
        tip = p0 * .55 + Vector((0, 0, .3 * s * (1.2 if big else 1)))
        mid = p0 * .95 + Vector((0, 0, .17 * s))
        mk(f'{e.name}_fang{k}', taper(spline([p0, mid, tip], 8), .055 * s, .003, 1.1, 10), IVORY(), e)
    for k in range(14 if big else 10):
        a = (k + .5) / (14 if big else 10) * math.tau; p0 = Vector((math.cos(a) * .23 * s, math.sin(a) * .23 * s, .02 * s))
        mk(f'{e.name}_tooth{k}', taper([p0, p0 * .85 + Vector((0, 0, .11 * s))], .03 * s, .002, 1, 8), IVORY(), e)
    tg = spline([(0, 0, .02 * s), (.05 * s, 0, .1 * s), (.18 * s if big else .1 * s, 0, .16 * s if big else .1 * s)], 10); N = len(tg)
    mk(e.name + '_tongue', loft(tg, [(.11 * s * (1 - i / N * .6), .03 * s) for i in range(N)], 12), TONGUE(), e)

def m_jaw():
    mouth_into(root('jaw'))

def m_maw():
    mouth_into(root('maw'), 1.3, True)

def m_horn():
    e = root('horn'); N = 28
    path = [Vector((.5 * (1 - math.cos(t * 1.9)), 0, .5 * math.sin(t * 1.9) + .02)) for t in (i / (N - 1) for i in range(N))]
    path[0] = Vector((0, 0, 0))
    mk('horn', loft(path, [.13 * (1 - i / N) ** .8 + .004 for i in range(N)], 14, prof=lambda t, a: 1 + .06 * math.sin(t * 40) * (1 - t)), IVORY(), e)
    mk('collar', torus(.14, .05, 20, 8, .6), BODY(), e)

def m_spike():
    e = root('spike')
    path = spline([(0, 0, -.03), (.02, 0, .2), (.09, 0, .46)], 14); N = len(path)
    mk('spike', loft(path, [.12 * (1 - i / N) ** 1.3 + .05 * (1 - i / N) ** 8 + .003 for i in range(N)], 12), IVORY(), e)

def m_frill():
    e = root('frill'); A = 14
    def fn(u, v):
        a = -1.35 + 2.7 * u; i = round(u * A)
        rr = .04 + v * (.78 if i % 2 == 0 else .66 + .12 * (abs(math.sin(u * A * math.pi / 2)) ** 2))
        return (math.sin(a) * rr, .04 * math.sin(u * 23) * rr + .1 * rr * rr, math.cos(a) * rr)
    mk('membrane', surface(fn, A * 2, 6), PATTERN(), e)
    for i in range(0, A + 1, 2):
        a = -1.35 + 2.7 * i / A
        path = [Vector((math.sin(a) * r, .1 * r * r + .012, math.cos(a) * r)) for r in (.02 + .76 * k / 7 for k in range(8))]
        mk(f'rib{i}', taper(path, .022, .006, 1, 8), BODY(), e)

def m_sac():
    e = root('sac')
    mk('sac', blob(.3, 3, (1, 1, .9), (0, 0, .14), .02, 6, 4), SAC(), e)

def m_antenna():
    e = root('antenna')
    path = spline([(0, 0, 0), (.05, 0, .4), (.18, 0, .75), (.3, 0, .86)], 16)
    mk('stalk', taper(path, .045, .018, 1, 10), BODY(), e)
    mk('bulb', blob(.08, 2, (1, 1, 1.2), (.31, 0, .88), .01, 12, 2), PATTERN(), e)

def m_fin():
    e = root('fin')
    def fn(u, v):
        bx = -.45 + .9 * u; tip = (-.5, 1.)
        x = bx * (1 - v) + tip[0] * v - .15 * v * v * u + .22 * math.sin(math.pi * v) * u
        z = tip[1] * v * (1 - .45 * u * (1 - v)) + .0
        return (x, .03 * math.sin(u * math.pi * 6) * v, z)
    mk('membrane', surface(fn, 12, 10), PATTERN(), e)
    for u in (.15, .4, .65, .9):
        path = [Vector(fn(u, v / 8)) + Vector((0, .01, 0)) for v in range(9)]
        mk(f'ray{u}', taper(path, .025, .005, 1, 8), BODY(), e)

def m_wing():
    e = root('wing')
    arm = spline([(0, 0, 0), (.05, 0, .5), (.15, 0, 1.)], 10)
    mk('arm', taper(arm, .1, .05, 1, 12), BODY(), e)
    hand = Vector((.15, 0, 1.)); tips = [(-.6, 0, 2.15), (-.12, 0, 2.5), (.35, 0, 2.35), (.72, 0, 1.85)]
    lines = []
    for k, t in enumerate(tips):
        mid = (hand + Vector(t)) * .5 + Vector((0, .06, 0))
        path = spline([hand, mid, t], 10); lines.append(path)
        mk(f'finger{k}', taper(path, .05, .01, 1, 8), BODY(), e)
    inner = spline([(0, 0, .05), (-.35, .05, .9), tips[0]], 10)
    edges = [inner] + lines
    bm = bmesh.new()
    for a, b in zip(edges, edges[1:]):
        va = [bm.verts.new(p + Vector((0, .02, 0))) for p in a]; vb = [bm.verts.new(p + Vector((0, .02, 0))) for p in b]
        for i in range(len(a) - 1): bm.faces.new((va[i], vb[i], vb[i + 1], va[i + 1]))
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=.002)
    mk('membrane', bm, MEMBRANE(), e)

def m_limb():
    e = root('limb')
    prof = [(.05, -.11), (.5, -.08), (.85, 0), (1.12, .3), (1.02, .55), (.78, .85), (.62, 1.), (.4, 1.07), (.05, 1.1)]
    mk('limb', lathe(prof, 14, 26, prof=lambda t, a: 1 + .12 * max(0, math.cos(a)) * math.sin(math.pi * t)), BODY(), e)

def foot_into(e, claws):
    pad = sphere(1, (.05, 0, .06), (.24, .2, .09))
    mk(e.name + '_pad', pad, BODY_DARK(), e)
    for k, y in enumerate((-.11, 0, .11)):
        mk(f'{e.name}_toe{k}', sphere(.07, (.25, y, .05), (1.3, 1, .8), 12, 8), BODY_DARK(), e)
        if claws:
            mk(f'{e.name}_claw{k}', taper(spline([(.3, y, .07), (.4, y * 1.1, .08), (.48, y * 1.15, .01)], 8), .045, .003, 1, 8), IVORY(), e)

def m_foot():
    foot_into(root('foot'), False)

def m_clawfoot():
    foot_into(root('clawfoot'), True)

# ---- cell parts
def m_mandible():
    e = root('mandible')
    path = spline([(0, 0, 0), (.02, 0, .25), (.12, 0, .45), (.26, 0, .5)], 16); N = len(path)
    mk('mandible', loft(path, [(.12 * (1 - i / N) ** .7 + .004, .06 * (1 - i / N) + .004) for i in range(N)], 12,
                        prof=lambda t, a: 1 + .12 * (math.sin(t * 34) > .3) * (math.cos(a) > .5)), IVORY(), e)

def m_frond():
    e = root('frond')
    mk('stalk', taper(spline([(0, 0, 0), (.02, 0, .15), (0, 0, .3)], 8), .03, .02, 1, 8), FROND(), e)
    leaf = spline([(0, 0, .26), (.01, 0, .4), (0, 0, .55)], 12); N = len(leaf)
    mk('leaf', loft(leaf, [(.1 * math.sin(math.pi * (i + .5) / N) ** .7, .025) for i in range(N)], 10), ALGAE(), e)

def m_nucleus():
    e = root('nucleus')
    mk('nucleus', blob(1, 3, (1, .9, .9), (0, 0, 0), .09, 2.5, 5, 3), PATTERN(), e)

def m_organelle():
    e = root('organelle')
    path = spline([(-.5, 0, 0), (0, .12, 0), (.5, 0, 0)], 14); N = len(path)
    mk('organelle', loft(path, [(.32 * math.sin(math.pi * (i + .3) / (N - .4)) ** .6 + .02, .22 * math.sin(math.pi * (i + .3) / (N - .4)) ** .6 + .02) for i in range(N)], 16,
                         prof=lambda t, a: 1 + .05 * math.sin(a * 7)), ORGANELLE(), e)

# ---------------------------------------------------------------- props
def m_tree_a():
    e = root('tree_a')
    trunk = spline([(0, 0, -.3), (.1, 0, 1.5), (-.1, .1, 3), (0, 0, 4.2)], 18)
    bm = loft(trunk, [.55 - .27 * i / 17 for i in range(18)], 16, prof=lambda t, a: 1 + .45 * max(0, 1 - t * 5) * abs(math.cos(a * 2.5)))
    sculpt(bm, .035, 4, 2, 2); mk('trunk', bm, BARK(), e)
    for k, d in enumerate([(1.4, .3, 4.6), (-1.2, .8, 4.4), (.2, -1.3, 4.8)]):
        mk(f'branch{k}', taper(spline([(0, 0, 3.2), (d[0] * .5, d[1] * .5, 3.9), d], 8), .2, .08, 1, 8), BARK(), e)
    for k, (p, r) in enumerate([((0, 0, 5.4), 1.9), ((1.5, .4, 4.9), 1.3), ((-1.4, .7, 4.8), 1.35), ((.3, -1.4, 5.0), 1.3), ((-.4, -.3, 6.4), 1.2), ((.9, .9, 5.9), 1.1)]):
        mk(f'leaves{k}', blob(r, 2, (1, 1, .85), p, .28, .9, k + 3), LEAF(), e)

def m_tree_b():
    e = root('tree_b')
    mk('trunk', taper(spline([(0, 0, -.3), (0, 0, 3), (0, 0, 6.5)], 10), .42, .12, 1, 12), BARK(), e)
    for k, (z0, R) in enumerate([(1.4, 2.3), (2.6, 1.9), (3.7, 1.5), (4.7, 1.1), (5.6, .7)]):
        bm = lathe([(.15, z0), (R, z0 - .15), (R * .75, z0 + .35), (R * .35, z0 + .95), (.02, z0 + 1.5)], 22, 14,
                   prof=lambda t, a: 1 + .14 * math.cos(a * 7) * (1 - t))
        sculpt(bm, .06, 2, k, 2); mk(f'tier{k}', bm, LEAF(), e)

def m_tree_c():
    e = root('tree_c')
    trunk = spline([(0, 0, -.2), (.4, 0, 2), (1.1, 0, 4), (1.3, 0, 5.4)], 26)
    mk('trunk', loft(trunk, [.36 - .1 * i / 25 for i in range(26)], 14, prof=lambda t, a: 1 + .07 * math.sin(t * 70)), BARK(), e)
    top = Vector((1.3, 0, 5.4))
    for k in range(8):
        a = k / 8 * math.tau + .3; d = Vector((math.cos(a), math.sin(a), 0))
        path = [top + d * (2.6 * t) + Vector((0, 0, .9 * t - 2.0 * t * t)) for t in (i / 13 for i in range(14))]
        mk(f'frond{k}', loft(path, [(.42 * math.sin(math.pi * (i + .4) / 13.4) ** .7, .03) for i in range(14)], 10), LEAF(), e)

def rock(name, seed, sc):
    e = root(name)
    bm = blob(1, 3, sc, (0, 0, 0), .35, .9, seed, 3)
    for v in bm.verts:
        if v.co.z < -.3: v.co.z = -.3 + (v.co.z + .3) * .2
    mk(name, bm, ROCK(), e)

def m_bush():
    e = root('bush')
    for k, (p, r) in enumerate([((0, 0, .4), .7), ((.6, .2, .3), .5), ((-.5, .3, .3), .55), ((.1, -.55, .3), .5), ((0, .1, .85), .45)]):
        mk(f'b{k}', blob(r, 2, (1, 1, .85), p, .12, 2, k), LEAF(), e)

def m_nest():
    e = root('nest', 2)
    bm = lathe([(.02, .05), (2.2, .08), (3.0, .45), (2.9, .65)], 28, 12); sculpt(bm, .06, 1.5, 4, 2)
    mk('bowl', bm, NEST_SOFT(), e)
    for k in range(70):
        c = random.uniform(0, math.tau); span = random.uniform(.5, .9); R = 3.3 + random.uniform(-.55, .55); z = random.uniform(.15, .85)
        path = [Vector((math.cos(c + span * t) * (R + .15 * math.sin(t * 3 + k)), math.sin(c + span * t) * (R + .15 * math.sin(t * 3 + k)), z + .18 * math.sin(t * math.pi + k))) for t in (i / 5 for i in range(6))]
        mk(f'twig{k}', taper(path, .085, .05, 1, 6), TWIG(), e)

def m_egg():
    e = root('egg')
    mk('egg', lathe([(.01, 0), (.42, .12), (.6, .55), (.48, 1.05), (.2, 1.38), (.01, 1.45)], 18, 16), EGG(), e)

def m_fruit():
    e = root('fruit')
    mk('fruit', lathe([(.02, .1), (.3, .02), (.5, .25), (.5, .55), (.35, .85), (.02, .76)], 18, 14), FRUIT(), e)
    mk('stem', taper(spline([(0, 0, .72), (.03, 0, .88), (.07, 0, 1.0)], 6), .03, .02, 1, 6), TWIG(), e)
    leaf = spline([(.05, 0, .95), (.2, .02, 1.04), (.38, 0, 1.02)], 10); N = len(leaf)
    mk('leaf', loft(leaf, [(.012, .1 * math.sin(math.pi * (i + .5) / N))for i in range(N)], 8), LEAF(), e)

def m_meat():
    e = root('meat')
    mk('meat', blob(1, 2, (.62, .45, .34), (0, 0, .3), .08, 3, 1, 2), MEAT(), e)
    mk('fat', blob(1, 2, (.5, .3, .1), (-.05, .1, .58), .03, 4, 2, 2), FAT(), e)
    mk('bone', taper([Vector((-.75, 0, .32)), Vector((.75, 0, .32))], .08, .08, 1, 10), IVORY(), e)
    for k, (x, y) in enumerate([(-.78, .07), (-.78, -.07), (.78, .07), (.78, -.07)]):
        mk(f'knob{k}', sphere(.11, (x, y, .32), u=12, v=8), IVORY(), e)

def m_plant():
    e = root('plant')
    for k, (p, r) in enumerate([((0, 0, 0), .5), ((.45, .2, .1), .35), ((-.4, .25, -.05), .38), ((.1, -.45, .05), .33), ((-.1, .1, .4), .3)]):
        mk(f'a{k}', blob(r, 2, (1, 1, .8), p, .12, 2.5, k), ALGAE(), e)

def m_meatbit():
    e = root('meatbit')
    mk('meat', blob(.6, 2, (1, .85, .75), (0, 0, 0), .15, 2.2, 7, 3), MEAT(), e)
    mk('fat', blob(.22, 1, (1, 1, .6), (.25, .2, .3), .03, 4, 2, 2), FAT(), e)

def m_relic_sea():
    e = root('relic_sea')
    mk('base', blob(.5, 2, (1.2, 1, .45), (0, 0, 0), .1, 2, 9, 2), STONE(), e)
    for k, (tilt, az, h) in enumerate([(0, 0, 1.3), (.45, 0, 1.0), (.5, 2.1, .9), (.4, 4.1, .8), (.7, 1.0, .6)]):
        bm = lathe([(.17, 0), (.2, h * .7), (.01, h)], 6, 6)
        r = rotm('Z', az) @ rotm('Y', tilt)
        mk(f'crystal{k}', xform(bm, (0, 0, .1), r), CRYSTAL(), e, smooth=False)

def m_relic_land():
    e = root('relic_land')
    mk('base', blob(.75, 2, (1.2, 1, .4), (0, 0, 0), .12, 2, 6, 2), STONE(), e)
    N = 60; path = []; radii = []
    for i in range(N):
        a = 1.0 + i / (N - 1) * 10.; R = .06 * math.exp(.19 * a)
        path.append(Vector((R * math.cos(a), 0, R * math.sin(a) + .78))); radii.append(.42 * R)
    mk('shell', loft(path, radii, 14, prof=lambda t, a: 1 + .07 * max(0, math.sin(t * 90)), cap0=True), FOSSIL(), e)
    mk('shard', xform(lathe([(.09, 0), (.1, .25), (.01, .42)], 6, 6), (.4, .3, .15), rotm('X', -.4)), CRYSTAL(), e, smooth=False)

# ================================================================ expansion parts
NOSE = lambda: mat('NOSE', '#e8909a', .4)
CHITIN = lambda: mat('CHITIN', '#4a3020', .3)
SPITM = lambda: mat('SPIT', '#9fe060', .2, emit=True)
HOOF = lambda: mat('HOOF', '#3a2e26', .5)
PAD = lambda: mat('PAD', '#d9849a', .6)
ANTLER = lambda: mat('ANTLER', '#d8c49a', .6)
ARMOR = lambda: mat('ARMOR', '#556070', .35, .2)
ELEC = lambda: mat('ELECTRIC', '#ffe45a', .2, emit=True)
TOXIN = lambda: mat('TOXIN', '#a060d0', .3)
SHELL = lambda: mat('SHELL', '#c8b48a', .4, double=True)
COMPOUND = lambda: mat('COMPOUND', '#7a1830', .1)
LANTERN = lambda: mat('LANTERN', '#fff07a', .2, emit=True)
EYESPOT = lambda: mat('EYESPOT', '#3060c0', .3, double=True)
REGEN = lambda: mat('REGEN', '#80f0a0', .1, alpha=.7)
GLOW = lambda: mat('GLOW', '#70ffe8', .2, emit=True)
FUR = lambda: mat('FUR', '#8a6a4a', .9)
BALEEN = lambda: mat('BALEEN', '#6e6458', .6, double=True)
CAMO = lambda: mat('CAMO', '#5a7a3a', .8, double=True)
V = Vector

def blade(path, width, thick=.012, segs=8, power=.7):
    """Flat leaf/feather/petal blade along a path (width peaks mid-way)."""
    N = len(path)
    return loft(path, [(width * math.sin(math.pi * (i + .5) / N) ** power + .004, thick) for i in range(N)], segs)

# ---- cell
def m_c_proboscis():
    e = root('c_proboscis')
    mk('base', sphere(.17, (0, 0, .02), (1, 1, .7)), BODY(), e)
    path = spline([(0, 0, 0), (.03, 0, .3), (.1, 0, .6), (.18, 0, .78)], 16); N = len(path)
    mk('tube', loft(path, [.09 - .04 * i / N for i in range(N)], 12, cap1=False), BODY(), e)
    mk('tip', xform(torus(.05, .022, 16, 8), (.18, 0, .78), rotm('Y', .6)), MOUTH(), e)

def m_c_jet():
    e = root('c_jet')
    mk('bulb', blob(.24, 3, (1, 1, .9), (0, 0, .14), .02, 5, 2), PATTERN(), e)
    mk('nozzle', lathe([(.13, .25), (.08, .42), (.1, .52), (.12, .56)], 16, 12), IVORY(), e)
    mk('hole', sphere(.08, (0, 0, .55), (1, 1, .25)), MOUTH(), e)

def m_c_electric():
    e = root('c_electric')
    mk('dome', sphere(.2, (0, 0, 0), (1, 1, .6)), BODY(), e)
    for k, z in enumerate((.12, .2, .28)):
        mk(f'coil{k}', xform(torus(.11 - .02 * k, .025, 20, 8), (0, 0, z)), ELEC(), e)
    for k in range(3):
        a = k / 3 * math.tau; p0 = V((math.cos(a) * .06, math.sin(a) * .06, .25)); tip = V((math.cos(a) * .18, math.sin(a) * .18, .5))
        mk(f'prong{k}', taper([p0, (p0 + tip) / 2 + V((0, 0, .03)), tip], .025, .008, 1, 8), IVORY(), e)
        mk(f'spark{k}', sphere(.04, tip, u=10, v=8), ELEC(), e)

def m_c_toxin():
    e = root('c_toxin')
    mk('sac', blob(.26, 3, (1, 1, 1.1), (0, 0, .2), .05, 5, 8, 3), TOXIN(), e)
    mk('spout', taper(spline([(0, 0, .42), (.05, 0, .55), (.12, 0, .62)], 8), .06, .03, 1, 10), BODY(), e)
    for k in range(5):
        a = k * 1.3
        mk(f'bump{k}', sphere(.05, (math.cos(a) * .22, math.sin(a) * .22, .12 + .06 * (k % 2)), u=10, v=8), SPITM(), e)

def m_c_shell():
    e = root('c_shell')
    def fn(u, v):
        a = (u - .5) * 1.6; b = (v - .5) * 1.3; r = .55; rid = .02 * math.sin(u * 28)
        return ((r + rid) * math.sin(a) * math.cos(b), (r + rid) * math.sin(b), (r + rid) * math.cos(a) * math.cos(b) - r * .8)
    mk('shell', surface(fn, 24, 14), SHELL(), e)

def m_c_tendril():
    e = root('c_tendril')
    path = spline([(0, 0, 0), (.05, 0, .3), (.15, 0, .55), (.35, 0, .65), (.45, 0, .5), (.35, 0, .4), (.28, 0, .48)], 30)
    mk('tendril', taper(path, .07, .012, 1, 10), BODY(), e)
    for k in range(6):
        p = path[3 + k * 4] + V((-.05, 0, 0))
        mk(f'sucker{k}', sphere(.025, p, u=8, v=6), PAD(), e)

# ---- mouths
def m_snout():
    e = root('snout')
    path = spline([(0, 0, 0), (.02, 0, .2), (.04, 0, .38)], 12); N = len(path)
    mk('snout', loft(path, [(.26 - .05 * i / N, .22 - .04 * i / N) for i in range(N)], 18, cap1=False), BODY(), e)
    mk('nose', sphere(1, (.04, 0, .38), (.2, .22, .05)), NOSE(), e)
    for k, y in enumerate((-.07, .07)):
        mk(f'nostril{k}', sphere(.045, (.04, y, .42), (1, 1, .4)), PUPIL(), e)

def m_mandibles():
    e = root('mandibles')
    mk('mouth', sphere(.2, (0, 0, 0), (1, 1, .35)), MOUTH(), e)
    for k, sd in enumerate((-1, 1)):
        path = spline([(0, sd * .16, 0), (0, sd * .24, .25), (0, sd * .13, .5), (0, -sd * .02, .58)], 16); N = len(path)
        mk(f'pincer{k}', loft(path, [(.04 * (1 - i / N) + .008, .075 * (1 - i / N) + .01) for i in range(N)], 10,
                             prof=lambda t, a: 1 + .15 * (math.sin(t * 30) > .4)), CHITIN(), e)

def m_spitter():
    e = root('spitter')
    mk('tube', lathe([(.21, 0), (.17, .2), (.13, .34)], 18, 10), BODY(), e)
    mk('lip', xform(torus(.13, .045, 22, 8), (0, 0, .35)), PATTERN(), e)
    mk('inside', sphere(.11, (0, 0, .36), (1, 1, .25)), MOUTH(), e)
    mk('glob', blob(.06, 2, (1, 1, 1), (0, 0, .43), .01, 8, 1, 2), SPITM(), e)

def m_sucker():
    e = root('sucker')
    mk('rim', torus(.28, .08, 28, 10, .8), BODY(), e)
    mk('disc', sphere(.27, (0, 0, -.01), (1, 1, .2)), MOUTH(), e)
    for ring, (R, n) in enumerate([(.22, 16), (.14, 10), (.07, 6)]):
        for k in range(n):
            a = (k + ring * .5) / n * math.tau; p0 = V((math.cos(a) * R, math.sin(a) * R, .03))
            mk(f'st{ring}_{k}', taper([p0, p0 * 1.05 + V((0, 0, .07))], .022, .002, 1, 6), IVORY(), e)

def m_tusks():
    e = root('tusks'); mouth_into(e, .9)
    for k, sd in enumerate((-1, 1)):
        path = spline([(.05, sd * .2, 0), (0, sd * .32, .25), (-.12, sd * .36, .45), (-.25, sd * .3, .55)], 16)
        mk(f'tusk{k}', taper(path, .08, .01, 1.1, 12), IVORY(), e)

def m_baleen():
    e = root('baleen')
    mk('lips', torus(.3, .06, 28, 8, .6), BODY(), e)
    mk('inside', sphere(.28, (0, 0, -.02), (1, 1, .25)), MOUTH(), e)
    for k in range(11):
        x = -.22 + .44 * k / 10; y = .04 * math.cos(x * 6)
        mk(f'plate{k}', loft([V((x, y, 0)), V((x, y, .15)), V((x * 1.05, y, .3))], [(.012, .05), (.012, .045), (.006, .02)], 6), BALEEN(), e)

# ---- eyes
def m_compound():
    e = root('compound')
    bm = blob(.32, 2, (1, 1, .75))
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < -.05], context='VERTS')
    mk('facets', bm, COMPOUND(), e, smooth=False)
    mk('rim', torus(.31, .04, 24, 8, .7), BODY(), e)

def m_beady():
    e = root('beady')
    mk('bead', sphere(.13, (0, 0, .06)), PUPIL(), e)
    mk('shine', sphere(.035, (.05, .04, .17)), EYE_W(), e)

def m_slit():
    e = root('slit'); eye_into(e, 1., pupil=(.22, 1, .55), iris=ELEC)

def m_goggle():
    e = root('goggle'); eye_into(e, 1.45, lid=False, iris=GLOW)
    mk('rim', xform(torus(.31, .07, 26, 10, .8), (0, 0, .12)), BODY_DARK(), e)

def m_triple():
    e = root('triple')
    mk('mound', sphere(.27, (0, 0, 0), (1, 1, .45)), BODY(), e)
    for x, y in [(-.1, -.1), (.12, -.06), (0, .13)]:
        eye_into(e, .5, V((x, y, .08)), lid=False)

def m_crystaleye():
    e = root('crystaleye')
    mk('socket', torus(.22, .07, 24, 8, .8), BODY(), e)
    mk('gem', lathe([(.01, -.05), (.2, .08), (.16, .2), (.01, .3)], 8, 8), CRYSTAL(), e, smooth=False)

# ---- limbs (feet for the leg rig + flipper)
def m_hoof():
    e = root('hoof')
    mk('hoof', lathe([(.01, 0), (.18, .01), (.16, .18), (.13, .24), (.01, .25)], 18, 10), HOOF(), e)
    mk('tuft', blob(.17, 2, (1, 1, .7), (0, 0, .27), .05, 6, 3, 2), BODY_DARK(), e)

def m_stompfoot():
    e = root('stompfoot')
    mk('pad', lathe([(.01, 0), (.34, .02), (.33, .2), (.24, .32), (.01, .34)], 22, 10), BODY_DARK(), e)
    for k, a in enumerate((-.5, -.17, .17, .5)):
        mk(f'nail{k}', sphere(1, (math.cos(a) * .32, math.sin(a) * .32, .08), (.06, .07, .05)), IVORY(), e)

def m_padfoot():
    e = root('padfoot')
    mk('paw', sphere(1, (.05, 0, .08), (.26, .22, .1)), BODY_DARK(), e)
    for k, (x, y) in enumerate([(.26, -.12), (.3, -.04), (.3, .04), (.26, .12)]):
        mk(f'bean{k}', sphere(.05, (x, y, .08), (1, 1, .8), 10, 8), PAD(), e)

def m_talonfoot():
    e = root('talonfoot'); foot_into(e, True)
    mk('sickle', taper(spline([(0, 0, .12), (.08, 0, .3), (.22, 0, .32), (.3, 0, .2)], 12), .06, .004, 1, 10), IVORY(), e)

def m_springfoot():
    e = root('springfoot')
    sole = spline([(-.1, 0, .04), (.2, 0, .03), (.5, 0, .02)], 10)
    mk('sole', loft(sole, [(.1, .035)] * 10, 12), BODY_DARK(), e)
    for k, y in enumerate((-.06, .06)):
        mk(f'toe{k}', taper(spline([(.45, y, .03), (.6, y * 1.5, .03), (.68, y * 1.8, 0)], 6), .03, .005, 1, 8), BODY_DARK(), e)

def m_flipper():
    e = root('flipper')
    def fn(u, v):
        w = .25 * (1 - v * .3) + .12 * math.sin(math.pi * v)
        return ((u - .5) * 2 * w - .25 * v * v, .02 * math.sin(u * math.pi), v)
    mk('paddle', surface(fn, 8, 12), BODY(), e)
    for k, u in enumerate((.2, .5, .8)):
        mk(f'bone{k}', taper([V(fn(u, v / 6)) + V((0, .015, 0)) for v in range(7)], .03, .01, 1, 6), BODY_DARK(), e)

# ---- weapons
def m_tailclub():
    e = root('tailclub')
    mk('handle', taper(spline([(0, 0, 0), (0, 0, .25), (0, 0, .4)], 6), .12, .1, 1, 12), BODY(), e)
    mk('club', blob(.3, 3, (1, 1, .85), (0, 0, .62), .06, 4, 11, 2), ARMOR(), e)
    for k in range(7):
        a = k / 6 * math.tau; d = V((math.cos(a), math.sin(a), .3)).normalized() if k < 6 else V((0, 0, 1))
        p0 = V((0, 0, .62)) + d * .25
        mk(f'spk{k}', taper([p0, p0 + d * .22], .06, .003, 1, 8), IVORY(), e)

def m_stinger():
    e = root('stinger')
    path = spline([(0, 0, 0), (.03, 0, .25), (.12, 0, .45), (.26, 0, .55)], 18); N = len(path)
    mk('tail', loft(path, [.12 - .05 * i / N for i in range(N)], 12, prof=lambda t, a: 1 + .18 * abs(math.sin(t * math.pi * 4))), BODY(), e)
    mk('bulb', blob(.11, 2, (1.2, 1, 1), (.3, 0, .57), .01, 6, 4, 2), PATTERN(), e)
    mk('sting', taper(spline([(.38, 0, .58), (.5, 0, .55), (.56, 0, .46)], 8), .04, .002, 1, 8), IVORY(), e)

def m_antlers():
    e = root('antlers')
    main = spline([(0, 0, 0), (-.05, 0, .4), (-.2, 0, .8), (-.45, 0, 1.05)], 18)
    mk('beam', taper(main, .08, .02, 1, 10), ANTLER(), e)
    for k, (i, d) in enumerate([(5, (.25, .1, .3)), (10, (.2, -.12, .35)), (14, (.1, .15, .3))]):
        p0 = main[i]
        mk(f'tine{k}', taper(spline([p0, p0 + V(d) * .6, p0 + V(d) + V((.05, 0, .1))], 8), .045, .01, 1, 8), ANTLER(), e)
    mk('burr', torus(.09, .035, 16, 8, .7), ANTLER(), e)

def m_armor():
    e = root('armor')
    for row in range(3):
        n = 4 - row % 2
        for k in range(n):
            x = (k - (n - 1) / 2) * .2; y = (row - 1) * .17
            mk(f'pl{row}{k}', xform(blob(1, 2, (.14, .12, .05)), (x, y, .05 + .03 * (1 - abs(x))), rotm('Y', -x * .6)), ARMOR(), e)

def m_quills():
    e = root('quills')
    mk('base', sphere(.18, (0, 0, 0), (1, 1, .5)), BODY(), e)
    for k in range(16):
        a = random.uniform(0, math.tau); tilt = random.uniform(.1, .7); L = random.uniform(.5, .85)
        d = V((math.sin(tilt) * math.cos(a), math.sin(tilt) * math.sin(a), math.cos(tilt))); p0 = d * .08
        mk(f'q{k}', taper([p0, p0 + d * L * .5, p0 + d * L * .8], .025, .01, 1, 6), IVORY(), e)
        mk(f'qt{k}', taper([p0 + d * L * .8, p0 + d * L], .01, .002, 1, 6), PUPIL(), e)

def m_shockorgan():
    e = root('shockorgan')
    mk('dome', sphere(.22, (0, 0, 0), (1, 1, .6)), BODY(), e)
    mk('orb', blob(.15, 2, (1, 1, 1), (0, 0, .25), .02, 8, 2, 2), ELEC(), e)
    for k, sd in enumerate((-1, 1)):
        mk(f'fin{k}', surface(lambda u, v, sd=sd: (sd * (.1 + .22 * v), 0, .05 + .35 * v * (1 - u * .5) + (u - .5) * .25 * (1 - v)), 4, 4), BODY(), e)
    for k in range(4):
        a = k / 4 * math.tau + .4; pts = [V((0, 0, .3))]
        for j in range(1, 5):
            r = .12 * j; pts.append(V((math.cos(a) * r + random.uniform(-.04, .04), math.sin(a) * r + random.uniform(-.04, .04), .3 + .06 * j)))
        mk(f'arc{k}', taper(pts, .015, .004, 1, 5), ELEC(), e, smooth=False)

# ---- social
def m_crest():
    e = root('crest')
    for k in range(5):
        a = -.55 + k * .275; d = V((-.35, math.sin(a), math.cos(a))).normalized(); L = .75 - abs(a) * .3
        path = [d * L * t + V((-.12 * t * t, 0, 0)) for t in (i / 10 for i in range(11))]
        mk(f'vane{k}', blade(path, .1), PATTERN(), e)
        mk(f'shaft{k}', taper(path, .018, .004, 1, 6), BODY(), e)

def m_plume():
    e = root('plume')
    for k, (sy, L) in enumerate([(-.25, 1.1), (0, 1.3), (.25, 1.1)]):
        path = [V((-1. * t * L, sy * t, .6 * t * L - .45 * t * t * L)) for t in (i / 14 for i in range(15))]
        mk(f'vane{k}', blade(path, .17, .012, 8, .5), PATTERN(), e)
        mk(f'shaft{k}', taper(path, .02, .004, 1, 6), BODY(), e)
        tip = path[11]
        mk(f'spot{k}', sphere(.08, tip + V((0, 0, .015)), (1.3, 1, .2), 12, 8), EYESPOT(), e)

def m_lantern():
    e = root('lantern')
    mk('stalk', taper(spline([(0, 0, 0), (.05, 0, .4), (.25, 0, .7), (.45, 0, .65)], 16), .04, .02, 1, 8), BODY(), e)
    mk('bulb', blob(.11, 2, (1, 1, 1.2), (.48, 0, .55), .01, 6, 2, 2), LANTERN(), e)
    mk('cap', sphere(.07, (.47, 0, .65), (1, 1, .5)), BODY(), e)

def m_dewlap():
    e = root('dewlap')
    def fn(u, v):
        a = u * math.pi; r = .1 + v * .55
        return (math.cos(a) * r * .9, .02 * math.sin(u * 14) * v, math.sin(a) * r)
    mk('flap', surface(fn, 16, 6), PATTERN(), e)
    for k in range(1, 5):
        a = k / 5 * math.pi
        mk(f'vein{k}', taper([V((math.cos(a) * r * .9, .012, math.sin(a) * r)) for r in (.1, .3, .5, .62)], .015, .005, 1, 6), BODY(), e)

def m_sail():
    e = root('sail'); tops = []
    for k in range(6):
        x = -.6 + k * .24; h = .5 + .5 * math.sin((k + .5) / 6 * math.pi); tops.append((x, h))
        mk(f'spine{k}', taper([V((x, 0, 0)), V((x - .05, 0, h * .5)), V((x - .1, 0, h))], .035, .008, 1, 6), IVORY(), e)
    def fn(u, v):
        kf = u * 5; i = min(int(kf), 4); f = kf - i; x0, h0 = tops[i]; x1, h1 = tops[i + 1]
        h = (h0 + (h1 - h0) * f) * (1 - .12 * math.sin(f * math.pi))
        return (x0 + (x1 - x0) * f - .1 * v, 0, v * h * .97)
    mk('membrane', surface(fn, 25, 6), MEMBRANE(), e)

def m_petals():
    e = root('petals')
    for k in range(8):
        a = k / 8 * math.tau; d = V((math.cos(a), math.sin(a), .9)).normalized()
        path = [d * (.05 + .5 * t) + V((0, 0, .08 * math.sin(t * math.pi))) for t in (i / 9 for i in range(10))]
        mk(f'p{k}', blade(path, .13, .015, 8, .6), PATTERN(), e)
    mk('center', blob(.1, 2, (1, 1, .7), (0, 0, .05), .02, 8, 2, 2), LANTERN(), e)

# ---- details
def m_camo():
    e = root('camo')
    for k in range(6):
        a = k * 1.1; r = .18 * (k % 3); c = V((math.cos(a) * r, math.sin(a) * r, .02)); d = V((math.cos(a + 1), math.sin(a + 1), 0))
        path = [c - d * .2 + d * .4 * t + V((0, 0, .03 * math.sin(t * math.pi))) for t in (i / 8 for i in range(9))]
        mk(f'leaf{k}', blade(path, .12, .01), CAMO(), e)

def m_regen():
    e = root('regen')
    for k, (p, r) in enumerate([((0, 0, .1), .16), ((.15, .08, .07), .11), ((-.12, .1, .06), .1), ((.02, -.15, .06), .12)]):
        mk(f'g{k}', blob(r, 2, (1, 1, 1), p, .01, 8, k, 2), REGEN(), e)

def m_glowspots():
    e = root('glowspots')
    for k in range(7):
        a = k * 2.4; r = .06 + .045 * k
        mk(f's{k}', sphere(.05 + .012 * (k % 3), (math.cos(a) * r, math.sin(a) * r, 0), (1, 1, .4), 12, 8), GLOW(), e)

def m_whiskers():
    e = root('whiskers')
    mk('pad', sphere(.12, (0, 0, 0), (1, 1, .5)), BODY(), e)
    for i, sd in enumerate((-1, 1)):
        for k in range(3):
            a = (k - 1) * .35
            path = spline([(0, sd * .05, .04), (math.cos(a) * .3, sd * (.25 + .1 * abs(a)), .12 + math.sin(a) * .1), (math.cos(a) * .6 - .1, sd * .6, .05 + math.sin(a) * .2)], 10)
            mk(f'w{i}{k}', taper(path, .012, .002, 1, 5), IVORY(), e)

def m_carapace():
    e = root('carapace')
    mk('shell', lathe([(.6, 0), (.62, .08), (.5, .28), (.28, .42), (.01, .47)], 28, 14, prof=lambda t, a: 1 + .04 * math.cos(a * 8)), SHELL(), e)
    for k, (R, z) in enumerate([(.56, .14), (.4, .33)]):
        mk(f'ridge{k}', xform(torus(R, .03, 32, 6, .8), (0, 0, z)), BODY_DARK(), e)

def m_mane():
    e = root('mane')
    for k in range(26):
        a = random.uniform(0, math.tau); rr = random.uniform(0, .25); p0 = V((math.cos(a) * rr, math.sin(a) * rr, 0))
        d = V((random.uniform(-.6, .2), random.uniform(-.3, .3), 1)).normalized(); L = random.uniform(.3, .55)
        mk(f'f{k}', taper(spline([p0, p0 + d * L * .5 + V((-.05, 0, 0)), p0 + d * L + V((-.15, 0, -.05))], 6), .05, .004, 1, 6), FUR(), e)

NEW_PARTS = [m_c_proboscis, m_c_jet, m_c_electric, m_c_toxin, m_c_shell, m_c_tendril,
             m_snout, m_mandibles, m_spitter, m_sucker, m_tusks, m_baleen,
             m_compound, m_beady, m_slit, m_goggle, m_triple, m_crystaleye,
             m_hoof, m_stompfoot, m_padfoot, m_talonfoot, m_springfoot, m_flipper,
             m_tailclub, m_stinger, m_antlers, m_armor, m_quills, m_shockorgan,
             m_crest, m_plume, m_lantern, m_dewlap, m_sail, m_petals,
             m_camo, m_regen, m_glowspots, m_whiskers, m_carapace, m_mane]

# ================================================================ jawed mouths, arms, more details
FUNGUS = lambda: mat('FUNGUS', '#e0803a', .5)
KERATIN = lambda: mat('KERATIN', '#3a3430', .35)

def jaw_half(name, L, W, D, side, nteeth, tooth, kind='cone', tusk=0., nostrils=False, tongue=False, beak=False, seed=0):
    """One half of a hinged jaw. Hinge at the origin, jaw reaching out along +Z.
    side=-1 builds the upper jaw (on -X), side=+1 the lower jaw; their flat biting
    faces meet near X=0 so the game can swing them open around the hinge."""
    e = root(name); N = 18; path = []; radii = []
    for i in range(N):
        t = i / (N - 1)
        path.append(V((side * D * .72 * (1 - .3 * t), 0, L * t)))
        tipf = math.sqrt(max(0., 1 - ((t - .78) / .22) ** 2)) if t > .78 else 1.
        radii.append((W * (1 - .3 * t * t) * tipf + .006, D * (1 - .45 * t) * tipf + .006))
    def prof(t, a):
        inner = max(0., -math.sin(a) * -side)          # loft frame y is -X
        return 1 - .55 * inner ** 1.5 + .03 * math.sin(a * 5 + t * 9) * (1 - inner)
    bm = loft(path, radii, 22, prof); sculpt(bm, .008, 6, seed, 2)
    mk(name + '_bone', bm, BODY(), e)
    face = lambda t: side * D * (.72 * (1 - .3 * t) - .45 * (1 - .45 * t))   # x of the flat biting face
    mk(name + '_gum', sphere(1, (face(.45) + side * .004, 0, L * .45), (D * .05, W * .7, L * .43)), MOUTH(), e)
    if tongue:
        tg = spline([(face(.1), 0, L * .08), (face(.4) - side * .02, 0, L * .4), (face(.7) - side * .03, 0, L * .62)], 10); n = len(tg)
        mk(name + '_tongue', loft(tg, [(W * .42 * (1 - i / n * .5), D * .12) for i in range(n)], 12), TONGUE(), e)
    for k in range(nteeth):
        t = (k + .7) / (nteeth + .4) * .84; z = L * t; ww = W * (1 - .3 * t * t) * .8
        for sy in (-1, 1):
            base = V((face(t), sy * ww, z))
            if kind == 'flat':
                mk(f'{name}_mol{k}{sy}', sphere(1, base + V((-side * tooth * .25, 0, 0)), (tooth * .45, tooth * .55, tooth * .6), 10, 8), IVORY(), e)
                continue
            ln = tooth * (1.25 if kind == 'saw' else 1) * (1 - .25 * t)
            tip = base + V((-side * ln, -sy * ww * .06, L * .02))
            mk(f'{name}_t{k}{sy}', taper([base - V((-side * .01, 0, 0)), base.lerp(tip, .5), tip], tooth * (.55 if kind == 'saw' else .38), .003, 1,
                                          3 if kind == 'saw' else 8), IVORY(), e, smooth=kind != 'saw')
    for sy in (-.4, .4):                                  # front incisors
        base = V((face(.93), sy * W * .5, L * .9))
        if kind == 'flat' and beak: continue
        mk(f'{name}_inc{sy}', taper([base, base + V((-side * tooth * .9, 0, L * .03))], tooth * .33, .003, 1, 8), IVORY(), e)
    if tusk:
        for sy in (-1, 1):
            base = V((face(.62), sy * W * .72, L * .62))
            mk(f'{name}_tusk{sy}', taper(spline([base, base + V((-side * tusk * .55, sy * .02, -.01)), base + V((-side * tusk, sy * .05, -.05))], 10), tusk * .2, .006, 1.1, 10), IVORY(), e)
    if nostrils:
        for sy in (-1, 1):
            c = V((side * D * 1.0, sy * W * .32, L * .9))
            mk(f'{name}_nb{sy}', sphere(1, c, (.03, .045, .05), 12, 8), BODY(), e)
            mk(f'{name}_nh{sy}', sphere(1, c + V((side * .022, 0, 0)), (.012, .02, .025), 8, 6), PUPIL(), e)
    if beak:
        tip = spline([(face(.75), 0, L * .75), (side * D * .5, 0, L * 1.02), (side * D * .1, 0, L * 1.08)], 10); n = len(tip)
        mk(name + '_beak', loft(tip, [(W * .55 * (1 - i / n) + .01, D * .5 * (1 - i / n) + .01) for i in range(n)], 14), KERATIN(), e)
    # scaly ridges along the outside of the jaw
    for k in range(5):
        t = .15 + k * .15; c = V((side * D * (.72 * (1 - .3 * t) + .42 * (1 - .45 * t)), 0, L * t))
        mk(f'{name}_sc{k}', sphere(1, c, (.012, W * .45 * (1 - .3 * t), .03)), BODY_DARK(), e)

def m_crocjaw():
    jaw_half('crocjaw_up', .9, .19, .13, -1, 8, .07, nostrils=True, seed=1)
    jaw_half('crocjaw_lo', .86, .17, .11, 1, 8, .065, tongue=True, seed=2)

def m_sawjaw():
    jaw_half('sawjaw_up', .5, .3, .13, -1, 6, .1, 'saw', seed=3)
    jaw_half('sawjaw_lo', .46, .28, .12, 1, 6, .095, 'saw', tongue=True, seed=4)

def m_crusher():
    jaw_half('crusher_up', .4, .32, .17, -1, 3, .07, seed=5)
    jaw_half('crusher_lo', .42, .34, .22, 1, 3, .06, tusk=.3, tongue=True, seed=6)

def m_grinder():
    jaw_half('grinder_up', .42, .27, .15, -1, 4, .09, 'flat', beak=True, seed=7)
    jaw_half('grinder_lo', .4, .25, .14, 1, 4, .09, 'flat', tongue=True, beak=True, seed=8)

# ---- hands for the arm rig: +Z runs along the forearm, fingers curl toward +X
def digit(name, base, ctrl, r0, r1, m, e, claw=0., cm=None):
    path = spline([base] + ctrl, 10)
    mk(name, taper(path, r0, r1, .8, 10), m, e)
    if claw:
        d = (path[-1] - path[-3]).normalized(); tip = path[-1] + d * claw
        hook = tip + V((claw * .5, 0, -claw * .15))
        mk(name + '_claw', taper(spline([path[-1] - d * .01, tip, hook], 8), r1 * 1.15, .003, 1, 8), cm or IVORY(), e)

def m_hand_grasp():
    e = root('hand_grasp')
    mk('palm', blob(1, 3, (.12, .15, .13), (0, 0, .1), .01, 5, 2, 2), BODY(), e)
    for k, y in enumerate((-.09, 0, .09)):
        digit(f'gf{k}', V((.01, y, .2)), [V((.04, y * 1.1, .33)), V((.12, y * 1.15, .42)), V((.22, y * 1.15, .42))], .042, .026, BODY(), e, .04)
    digit('gthumb', V((.05, .12, .08)), [V((.12, .2, .14)), V((.2, .2, .2))], .042, .028, BODY(), e, .035)

def m_hand_claw():
    e = root('hand_claw')
    mk('paw', blob(1, 3, (.15, .19, .14), (0, 0, .1), .015, 5, 3, 2), BODY_DARK(), e)
    mk('tuft', blob(.13, 2, (1, 1.2, .6), (-.05, 0, .02), .05, 7, 4, 2), FUR(), e)
    for k, y in enumerate((-.11, 0, .11)):
        digit(f'cf{k}', V((.03, y, .2)), [V((.07, y * 1.1, .26))], .05, .04, BODY_DARK(), e)
        p0 = V((.08, y * 1.1, .27))
        mk(f'rake{k}', taper(spline([p0, p0 + V((.04, 0, .15)), p0 + V((.17, 0, .27)), p0 + V((.31, 0, .24))], 14), .045, .003, 1.1, 10), IVORY(), e)

def m_hand_scythe():
    e = root('hand_scythe')
    mk('knuckle', blob(1, 3, (.1, .09, .12), (0, 0, .06), .01, 6, 5, 2), CHITIN(), e)
    path = spline([(0, 0, .1), (.02, 0, .4), (.12, 0, .66), (.3, 0, .72), (.42, 0, .62)], 26); N = len(path)
    mk('blade', loft(path, [(.018, .07 * math.sin(math.pi * min(1, (i + 2) / N)) ** .6 + .006) for i in range(N)], 10,
                      prof=lambda t, a: 1 + .25 * max(0, -math.sin(a)) * (math.sin(t * 70) > 0)), CHITIN(), e)
    for k in range(6):
        b = path[4 + k * 3]; mk(f'spine{k}', taper([b + V((.045, 0, 0)), b + V((.1, 0, .03))], .015, .002, 1, 6), IVORY(), e)

def m_hand_pincer():
    e = root('hand_pincer')
    mk('propodus', blob(1, 3, (.16, .11, .24), (0, 0, .22), .02, 5, 6, 2), PATTERN(), e)
    for k in range(4):
        mk(f'knob{k}', sphere(.025, (-.12 + .02 * k, .07 * (-1) ** k, .14 + .07 * k), (1, 1, .8), 10, 8), BODY_DARK(), e)
    digit('fixed', V((-.06, 0, .4)), [V((-.07, 0, .55)), V((-.03, 0, .72)), V((.04, 0, .8))], .07, .015, PATTERN(), e)
    digit('dactyl', V((.08, 0, .38)), [V((.17, 0, .5)), V((.17, 0, .66)), V((.1, 0, .76))], .065, .015, PATTERN(), e)
    for k in range(4):
        mk(f'pt{k}', sphere(.018, (-.04 + .005 * k, 0, .5 + .06 * k), (1.4, 1, 1), 8, 6), IVORY(), e)
        mk(f'dt{k}', sphere(.018, (.12 - .01 * k, 0, .5 + .06 * k), (1.4, 1, 1), 8, 6), IVORY(), e)

def m_hand_shovel():
    e = root('hand_shovel')
    mk('palm', blob(1, 3, (.09, .24, .17), (0, 0, .14), .012, 5, 7, 2), BODY_DARK(), e)
    mk('palmpad', sphere(1, (.07, 0, .16), (.03, .17, .12)), PAD(), e)
    for k in range(5):
        y = (k - 2) * .09; b = V((.0, y, .28))
        mk(f'dig{k}', loft(spline([b, b + V((.03, y * .1, .12)), b + V((.08, y * .15, .2))], 10), [(.034 * (1 - i / 11) + .012, .016) for i in range(10)], 8), IVORY(), e)

def m_tentseg():
    e = root('tentseg')
    mk('seg', lathe([(.01, -.05), (.95, 0), (1., .5), (.9, 1.), (.01, 1.05)], 14, 14), BODY(), e)
    for k, z in enumerate((.28, .72)):
        mk(f'suck{k}', sphere(1, (.85, 0, z), (.18, .42, .2), 12, 8), PAD(), e)
        mk(f'suckh{k}', sphere(1, (.98, 0, z), (.06, .18, .09), 8, 6), MOUTH(), e)

# ---- more details
def m_gills():
    e = root('gills')
    for k in range(4):
        y = (k - 1.5) * .11
        ridge = spline([(-.22, y, 0), (0, y + .01, .05), (.22, y, 0)], 14)
        mk(f'gr{k}', loft(ridge, [(.03, .025)] * 14, 10), BODY(), e)
        mk(f'gs{k}', sphere(1, (0, y + .045, .02), (.2, .016, .022)), MOUTH(), e)
        for j in range(5):
            x = -.14 + j * .07; b = V((x, y + .045, .02))
            mk(f'gf{k}{j}', taper(spline([b, b + V((0, .02, .04)), b + V((0, .05, .05))], 6), .012, .003, 1, 6), TONGUE(), e)

def m_barnacles():
    e = root('barnacles'); rnd = random.Random(21)
    for k in range(7):
        a = k * 2.3; r = .07 + .04 * k; c = V((math.cos(a) * r * .9, math.sin(a) * r * .9, 0)); sc = .7 + rnd.random() * .6
        bm = lathe([(.085 * sc, 0), (.08 * sc, .03 * sc), (.05 * sc, .1 * sc), (.035 * sc, .115 * sc)], 10, 8,
                   prof=lambda t, a: 1 + .12 * math.cos(a * 6))
        mk(f'cone{k}', xform(bm, c), SHELL(), e, smooth=False)
        mk(f'hole{k}', sphere(.03 * sc, c + V((0, 0, .11 * sc)), (1, 1, .4), 10, 6), MOUTH(), e)

def m_scales():
    e = root('scales')
    for i in range(5):
        for j in range(4):
            x = -.24 + i * .12; y = -.18 + j * .12 + (i % 2) * .06
            if abs(y) > .23: continue
            bm = sphere(1, (0, 0, 0), (.075, .065, .016), 14, 8); xform(bm, (x, y, .02 + .01 * math.cos(x * 6)), rotm('Y', .38))
            mk(f'sc{i}{j}', bm, PATTERN() if (i + j) % 3 else BODY_DARK(), e)

def m_warts():
    e = root('warts')
    for k in range(8):
        a = k * 2.4; r = .05 + .035 * k; c = V((math.cos(a) * r, math.sin(a) * r, 0)); s = .05 + .015 * (k % 3)
        mk(f'w{k}', blob(s, 2, (1, 1, .8), c, s * .2, 9, k, 2), BODY_DARK(), e)
        mk(f'wt{k}', sphere(s * .38, c + V((0, 0, s * .75)), (1, 1, .6), 10, 6), SPITM(), e)

def m_eyespots():
    e = root('eyespots')
    for k, (x, s) in enumerate(((-.12, 1.), (.17, .7))):
        mk(f'o{k}', sphere(1, (x, 0, 0), (.2 * s, .16 * s, .02)), PATTERN(), e)
        mk(f'r{k}', sphere(1, (x, 0, .012), (.15 * s, .12 * s, .02)), PUPIL(), e)
        mk(f'i{k}', sphere(1, (x, 0, .022), (.1 * s, .085 * s, .02)), EYESPOT(), e)
        mk(f'h{k}', sphere(1, (x - .04 * s, .03 * s, .035), (.028 * s, .028 * s, .012)), EYE_W(), e)

def m_fungus():
    e = root('fungus')
    for k, (x, y, h, r) in enumerate([(0, 0, .22, .13), (.14, .06, .14, .08), (-.12, .09, .12, .07), (.06, -.13, .1, .06), (-.08, -.1, .16, .09)]):
        mk(f'st{k}', xform(lathe([(r * .45, 0), (r * .3, h * .5), (r * .28, h)], 10, 8), (x, y, 0)), FAT(), e)
        mk(f'cap{k}', xform(lathe([(r * .3, h * .9), (r, h * .95), (r * .9, h * 1.15), (r * .4, h * 1.32), (.002, h * 1.36)], 16, 10), (x, y, 0)),
           FUNGUS() if k % 2 == 0 else GLOW(), e)
        if k % 2 == 0:
            for j in range(4):
                a = j * 1.6 + k; mk(f'dot{k}{j}', sphere(r * .13, (x + math.cos(a) * r * .55, y + math.sin(a) * r * .55, h * 1.22), (1, 1, .5), 8, 6), IVORY(), e)

def m_crystals():
    e = root('crystals'); rnd = random.Random(5)
    mk('base', blob(.15, 2, (1.3, 1.1, .45), (0, 0, 0), .03, 6, 9, 2), STONE(), e)
    for k in range(6):
        h = .2 + rnd.random() * .3; r = .045 + rnd.random() * .03
        bm = lathe([(.002, 0), (r, .03), (r, h * .8), (.002, h)], 6, 4)
        a = k / 6 * math.tau; tilt = .25 + rnd.random() * .3
        xform(bm, (math.cos(a) * .08, math.sin(a) * .07, .02), rotm(V((-math.sin(a), math.cos(a), 0)), tilt))
        mk(f'cr{k}', bm, CRYSTAL(), e, smooth=False)

def m_ears():
    e = root('ears')
    def fn(u, v):
        w = .17 * math.sin(math.pi * min(1, v * 1.15 + .08)) ** .7
        x = (u - .5) * 2 * w; return (-.12 * v * v + .05 * ((u - .5) * 2) ** 2, x, v * .55)
    mk('ear', surface(fn, 10, 12), BODY(), e)
    def fi(u, v):
        x, y, z = fn(.15 + u * .7, .1 + v * .75); return (x + .012, y, z)
    mk('inner', surface(fi, 8, 10), PATTERN(), e)
    mk('tip', blob(.04, 2, (1, 1, 1.6), (-.12, 0, .56), .015, 9, 2, 2), FUR(), e)
    mk('earbase', sphere(1, (0, 0, .02), (.08, .12, .05)), BODY(), e)

NEW2_PARTS = [m_crocjaw, m_sawjaw, m_crusher, m_grinder,
              m_hand_grasp, m_hand_claw, m_hand_scythe, m_hand_pincer, m_hand_shovel, m_tentseg,
              m_gills, m_barnacles, m_scales, m_warts, m_eyespots, m_fungus, m_crystals, m_ears]

# ---------------------------------------------------------------- build
for f in NEW_PARTS + NEW2_PARTS + [m_eye, m_stalkeye, m_beak, m_jaw, m_maw, m_horn, m_spike, m_frill, m_sac, m_antenna, m_fin, m_wing,
          m_limb, m_foot, m_clawfoot, m_mandible, m_frond, m_nucleus, m_organelle,
          m_tree_a, m_tree_b, m_tree_c, lambda: rock('rock_a', 1, (1.3, 1, .8)), lambda: rock('rock_b', 5, (1, 1.2, 1.1)),
          m_bush, m_nest, m_egg, m_fruit, m_meat, m_plant, m_meatbit, m_relic_sea, m_relic_land]:
    f()

blend = os.path.join(HERE, 'primordia_models.blend')
glb = os.path.join(HERE, 'primordia_parts.glb')
bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.export_scene.gltf(filepath=glb, export_format='GLB', export_texcoords=False, export_normals=True,
                          export_yup=True, export_apply=True, export_cameras=False, export_lights=False)
tris = sum(len(o.data.polygons) for o in bpy.data.objects if o.type == 'MESH')
print('PRIMORDIA_OK objects=%d faces=%d glb=%d bytes' % (len(bpy.data.objects), tris, os.path.getsize(glb)))

# ---------------------------------------------------------------- preview render
cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')); COL.objects.link(cam)
cam.location = (15.75, -40, 30); cam.rotation_euler = (math.radians(52), 0, 0); cam.data.lens = 30
SC.camera = cam
sun = bpy.data.objects.new('Sun', bpy.data.lights.new('Sun', 'SUN')); COL.objects.link(sun); sun.rotation_euler = (.6, .2, .4)
SC.render.engine = 'BLENDER_WORKBENCH'
SC.display.shading.color_type = 'MATERIAL'
SC.display.shading.light = 'STUDIO'
SC.render.resolution_x, SC.render.resolution_y = 1600, 1200
SC.render.filepath = os.path.join(HERE, 'preview.png')
bpy.ops.render.render(write_still=True)
# close-up of the creature parts (first two rows of the layout)
SHEETS = {
    'preview_cell_mouths_eyes': ['c_proboscis', 'c_jet', 'c_electric', 'c_toxin', 'c_shell', 'c_tendril', 'mandible', 'frond',
                                 'beak', 'jaw', 'maw', 'snout', 'mandibles', 'spitter', 'sucker', 'tusks',
                                 'baleen', 'eye', 'stalkeye', 'compound', 'beady', 'slit', 'goggle', 'triple', 'crystaleye'],
    'preview_limbs_weapons': ['limb', 'foot', 'clawfoot', 'hoof', 'stompfoot', 'padfoot', 'talonfoot', 'springfoot',
                              'wing', 'flipper', 'horn', 'spike', 'tailclub', 'stinger', 'antlers', 'armor', 'quills', 'shockorgan'],
    'preview_jaws_arms': ['crocjaw_up', 'crocjaw_lo', 'sawjaw_up', 'sawjaw_lo', 'crusher_up', 'crusher_lo', 'grinder_up', 'grinder_lo',
                          'hand_grasp', 'hand_claw', 'hand_scythe', 'hand_pincer', 'hand_shovel', 'tentseg',
                          'gills', 'barnacles', 'scales', 'warts', 'eyespots', 'fungus', 'crystals', 'ears'],
    'preview_social_details': ['frill', 'sac', 'crest', 'plume', 'lantern', 'dewlap', 'sail', 'petals',
                               'antenna', 'fin', 'camo', 'regen', 'glowspots', 'whiskers', 'carapace', 'mane'],
}
home = {o.name: o.location.copy() for o in bpy.data.objects if o.name.startswith('P_') and o.parent is None}
for sheet, names in SHEETS.items():
    for n, loc in home.items():
        bpy.data.objects[n].location = loc + V((0, 0, -500))
    for i, n in enumerate(names):
        bpy.data.objects['P_' + n].location = ((i % 8) * 1.7, -(i // 8) * 2.2, 0)
    rows = (len(names) + 7) // 8
    cam.location = (5.95, -2.2 * (rows - 1) / 2 - 9.5, 7.0); cam.rotation_euler = (math.radians(58), 0, 0); cam.data.lens = 30
    SC.render.resolution_x, SC.render.resolution_y = 1600, 1000
    SC.render.filepath = os.path.join(HERE, sheet + '.png')
    bpy.ops.render.render(write_still=True)
print('PRIMORDIA_RENDER_OK')
