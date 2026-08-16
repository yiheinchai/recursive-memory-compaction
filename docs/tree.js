/* The memory tree, animated.
 *
 * This is the whole thesis in one picture, so it is worth the code: a store
 * that GROWS downward in knowledge while the band at the top — what recall
 * actually loads on every prompt — SHRINKS. Every other memory system moves
 * those two together.
 *
 * Geometry is hand-placed rather than laid out, because a stable column per
 * lesson is what lets the eye follow one lesson climbing and narrowing across
 * states. Widths are token counts, to scale. Nothing here is decorative.
 */
(function () {
  "use strict";

  var SVGNS = "http://www.w3.org/2000/svg";
  var SCALE = 0.5;                 // px per token
  var ROW = { 2: 220, 1: 300, 0: 380 };
  var NODE_H = 34;
  var BAND = { x: 56, y: 36, w: 800, h: 48, pad: 20, gap: 12, barY: 49, barH: 22 };

  // Every node the story ever contains. `col` is the visual column it lives in
  // so a lesson stays put as it is compressed.
  var NODES = {
    A0: { tok: 260, level: 0, cx: 190 },
    B0: { tok: 220, level: 0, cx: 450 },
    C0: { tok: 200, level: 0, cx: 700 },
    A1: { tok: 195, level: 1, cx: 190 },
    B1: { tok: 165, level: 1, cx: 450 },
    A2: { tok: 146, level: 2, cx: 190 },
    M:  { tok: 210, level: 2, cx: 575 }
  };

  var COLUMNS = [
    { cx: 190, label: "retrying flaky calls" },
    { cx: 450, label: "cache invalidation" },
    { cx: 700, label: "deploy rollback" }
  ];

  // Each state is the entire store at one moment.
  var STATES = [
    {
      nodes: [], edges: [], load: [], used: [], kept: 0,
      cap: "A fresh repo. RMC knows nothing yet, and costs nothing."
    },
    {
      nodes: ["A0"], edges: [], load: ["A0"], used: [], kept: 1,
      cap: "You correct the agent once. The lesson is written down verbose and exact — nothing is summarised away at capture."
    },
    {
      nodes: ["A0", "B0", "C0"], edges: [], load: ["A0", "B0", "C0"], used: [], kept: 3,
      cap: "More sessions, more lessons. This is where every other memory system stops, and where the bill starts growing forever."
    },
    {
      nodes: ["A0", "B0", "C0", "A1"], edges: [["A0", "A1"]],
      load: ["A1", "B0", "C0"], used: ["A1"], kept: 3,
      cap: "The first lesson was loaded, actually used, and the work succeeded. That is evidence it can be said shorter — so it earns a compressed form."
    },
    {
      nodes: ["A0", "B0", "C0", "A1", "B1", "A2"],
      edges: [["A0", "A1"], ["A1", "A2"], ["B0", "B1"]],
      load: ["A2", "B1", "C0"], used: ["A2", "B1"], kept: 3,
      cap: "Used again, shortened again. The verbose originals are still down there — compression adds a level above, it never deletes."
    },
    {
      nodes: ["A0", "B0", "C0", "A1", "B1", "A2", "M"],
      edges: [["A0", "A1"], ["A1", "A2"], ["B0", "B1"], ["B1", "M"], ["C0", "M"]],
      load: ["A2", "M"], used: ["M"], kept: 3,
      cap: "Two lessons keep being needed together, so they earn a shared parent. That is abstraction across lessons, not just compression within one."
    },
    {
      nodes: ["A0", "B0", "C0", "A1", "B1", "A2", "M"],
      edges: [["A0", "A1"], ["A1", "A2"], ["B0", "B1"], ["B1", "M"], ["C0", "M"]],
      load: ["A2", "M"], used: [], kept: 3, patch: 38, descend: ["A2", "A1"],
      cap: "And when the short form is not enough, the manifest names which dropped detail to pull back up — so you pay for depth only in the moment you need it."
    }
  ];

  function el(name, attrs) {
    var n = document.createElementNS(SVGNS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function width(id) { return NODES[id].tok * SCALE; }

  function target(state) {
    // Where every node wants to be, and how visible, in this state.
    var out = {};
    for (var id in NODES) {
      var n = NODES[id];
      out[id] = {
        x: n.cx - width(id) / 2,
        y: ROW[n.level],
        w: width(id),
        o: state.nodes.indexOf(id) >= 0 ? 1 : 0
      };
    }
    // Pack the context band left to right, in load order.
    var x = BAND.x + BAND.pad;
    out._band = {};
    for (var i = 0; i < state.load.length; i++) {
      var w = width(state.load[i]);
      out._band[state.load[i]] = { x: x, w: w };
      x += w + BAND.gap;
    }
    out._patch = state.patch ? { x: x, w: state.patch * SCALE } : null;
    return out;
  }

  function build(root) {
    var svg = el("svg", {
      viewBox: "0 0 900 470", fill: "none", stroke: "#1a1a1a",
      "stroke-width": "1.5", role: "img",
      "aria-label": "A memory tree growing downward while the context loaded on every prompt shrinks"
    });

    var chrome = el("g", {});
    chrome.appendChild(el("rect", {
      x: BAND.x, y: BAND.y, width: BAND.w, height: BAND.h,
      "stroke-dasharray": "5 5", opacity: ".55"
    }));
    var labels = el("g", {
      stroke: "none", fill: "#1a1a1a",
      "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace"
    });
    function txt(x, y, s, size, op, anchor) {
      var t = el("text", { x: x, y: y, "font-size": size || 13, opacity: op || 1 });
      if (anchor) t.setAttribute("text-anchor", anchor);
      t.textContent = s;
      labels.appendChild(t);
      return t;
    }
    txt(BAND.x, 26, "LOADED INTO CONTEXT, EVERY PROMPT", 11.5, 0.6).setAttribute("letter-spacing", "1.4");
    txt(BAND.x, 130, "THE STORE", 11.5, 0.6).setAttribute("letter-spacing", "1.4");
    [2, 1, 0].forEach(function (lv) {
      txt(22, ROW[lv] + 22, "L" + lv, 13, 0.45);
    });
    COLUMNS.forEach(function (c) { txt(c.cx, 432, c.label, 12, 0.55, "middle"); });
    chrome.appendChild(labels);

    var gEdges = el("g", { opacity: ".45" });
    var gNodes = el("g", {});
    var gBand = el("g", {});
    svg.appendChild(chrome);
    svg.appendChild(gEdges);
    svg.appendChild(gNodes);
    svg.appendChild(gBand);

    var parts = {};
    for (var id in NODES) {
      var g = el("g", {});
      var rect = el("rect", { height: NODE_H, y: 0, x: 0, width: 0, fill: "#f1f1f1" });
      var label = el("text", {
        "font-size": 12.5, "text-anchor": "middle", stroke: "none",
        fill: "#1a1a1a", "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace"
      });
      var badge = el("path", { "stroke-width": 2.2, "stroke-linecap": "round", "stroke-linejoin": "round", opacity: 0 });
      g.appendChild(rect); g.appendChild(label); g.appendChild(badge);
      gNodes.appendChild(g);
      parts[id] = { g: g, rect: rect, label: label, badge: badge };
    }

    var bandBars = {};
    for (var id2 in NODES) {
      var b = el("rect", { y: BAND.barY, height: BAND.barH, x: 0, width: 0, fill: "#1a1a1a", stroke: "none", opacity: 0 });
      gBand.appendChild(b);
      bandBars[id2] = b;
    }
    var patchBar = el("rect", {
      y: BAND.barY, height: BAND.barH, x: 0, width: 0,
      fill: "none", stroke: "#1a1a1a", "stroke-dasharray": "3 3", opacity: 0
    });
    gBand.appendChild(patchBar);

    root.appendChild(svg);
    return { svg: svg, parts: parts, gEdges: gEdges, bandBars: bandBars, patchBar: patchBar };
  }

  function init(root) {
    var hud = {
      kept: root.querySelector("[data-tree-kept]"),
      tok: root.querySelector("[data-tree-tok]"),
      cap: root.querySelector("[data-tree-cap]"),
      dots: root.querySelectorAll("[data-tree-step]")
    };
    var view = build(root.querySelector("[data-tree-canvas]"));

    var cur = {};
    for (var id in NODES) cur[id] = { x: NODES[id].cx, y: ROW[NODES[id].level], w: 0, o: 0 };
    var curBand = {}, curPatch = { x: 0, w: 0, o: 0 };
    for (var id2 in NODES) curBand[id2] = { x: BAND.x + BAND.pad, w: 0, o: 0 };

    var index = 0, from = null, to = target(STATES[0]), t0 = 0, DUR = 700, raf = null;
    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function ease(t) { return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }
    function lerp(a, b, k) { return a + (b - a) * k; }

    function paint() {
      var s = STATES[index];
      for (var id in NODES) {
        var c = cur[id], p = view.parts[id];
        p.g.setAttribute("transform", "translate(" + c.x + "," + c.y + ")");
        p.g.setAttribute("opacity", c.o);
        p.rect.setAttribute("width", Math.max(0, c.w));
        p.rect.setAttribute("stroke-width", s.load.indexOf(id) >= 0 ? 2.6 : 1.5);
        p.label.setAttribute("x", c.w / 2);
        p.label.setAttribute("y", NODE_H / 2 + 4.5);
        p.label.textContent = c.w > 46 ? NODES[id].tok : "";
        var isUsed = s.used.indexOf(id) >= 0;
        p.badge.setAttribute("opacity", isUsed ? 1 : 0);
        p.badge.setAttribute("d", isUsed
          ? "M" + (c.w + 10) + " " + (NODE_H / 2) + "l5 5 9 -11" : "M0 0");
      }

      // Edges are redrawn from live positions so they track the tween.
      while (view.gEdges.firstChild) view.gEdges.removeChild(view.gEdges.firstChild);
      s.edges.forEach(function (e) {
        var a = cur[e[0]], b = cur[e[1]];
        var hot = s.descend && s.descend.indexOf(e[0]) >= 0 && s.descend.indexOf(e[1]) >= 0;
        // Diagonal, not vertical: a merge parent sits in its own column, so a
        // vertical stub would stop short of it.
        var line = el("path", {
          d: "M" + (a.x + a.w / 2) + " " + a.y +
             "L" + (b.x + b.w / 2) + " " + (b.y + NODE_H),
          stroke: "#1a1a1a", "stroke-width": hot ? 2.6 : 1.5,
          opacity: hot ? 1 : 0.55, "stroke-dasharray": hot ? "6 4" : ""
        });
        view.gEdges.appendChild(line);
      });

      for (var id3 in NODES) {
        var cb = curBand[id3], bar = view.bandBars[id3];
        bar.setAttribute("x", cb.x);
        bar.setAttribute("width", Math.max(0, cb.w));
        bar.setAttribute("opacity", cb.o);
      }
      view.patchBar.setAttribute("x", curPatch.x);
      view.patchBar.setAttribute("width", Math.max(0, curPatch.w));
      view.patchBar.setAttribute("opacity", curPatch.o);

      var loaded = s.load.reduce(function (n, id) { return n + NODES[id].tok; }, 0) + (s.patch || 0);
      hud.kept.textContent = s.kept;
      hud.tok.textContent = loaded;
      hud.cap.textContent = s.cap;
      Array.prototype.forEach.call(hud.dots, function (d, i) {
        d.setAttribute("aria-selected", String(i === index));
      });
    }

    function step(now) {
      var k = reduced ? 1 : ease(Math.min(1, (now - t0) / DUR));
      for (var id in NODES) {
        cur[id].x = lerp(from[id].x, to[id].x, k);
        cur[id].y = lerp(from[id].y, to[id].y, k);
        cur[id].w = lerp(from[id].w, to[id].w, k);
        cur[id].o = lerp(from[id].o, to[id].o, k);
        var fb = from._band[id], tb = to._band[id];
        var b = curBand[id];
        b.x = lerp(fb ? fb.x : b.x, tb ? tb.x : b.x, k);
        b.w = lerp(fb ? fb.w : 0, tb ? tb.w : 0, k);
        b.o = lerp(fb ? 1 : 0, tb ? 1 : 0, k);
      }
      curPatch.x = to._patch ? to._patch.x : curPatch.x;
      curPatch.w = lerp(from._patch ? from._patch.w : 0, to._patch ? to._patch.w : 0, k);
      curPatch.o = lerp(from._patch ? 1 : 0, to._patch ? 1 : 0, k);
      paint();
      if (k < 1) raf = requestAnimationFrame(step);
    }

    function go(i) {
      index = ((i % STATES.length) + STATES.length) % STATES.length;
      from = {};
      for (var id in NODES) from[id] = { x: cur[id].x, y: cur[id].y, w: cur[id].w, o: cur[id].o };
      from._band = {};
      for (var id2 in NODES) if (curBand[id2].o > 0.01) from._band[id2] = { x: curBand[id2].x, w: curBand[id2].w };
      from._patch = curPatch.o > 0.01 ? { x: curPatch.x, w: curPatch.w } : null;
      to = target(STATES[index]);
      t0 = performance.now();
      if (raf) cancelAnimationFrame(raf);
      raf = requestAnimationFrame(step);
    }

    var timer = null;
    function play() { stop(); timer = setInterval(function () { go(index + 1); }, 3400); }
    function stop() { if (timer) clearInterval(timer); timer = null; }

    Array.prototype.forEach.call(hud.dots, function (d, i) {
      d.addEventListener("click", function () { stop(); go(i); });
    });
    root.addEventListener("mouseenter", stop);
    root.addEventListener("mouseleave", function () { if (!reduced) play(); });

    go(0);
    if (!reduced) {
      // Only run while it is actually on screen.
      new IntersectionObserver(function (entries) {
        entries[0].isIntersecting ? play() : stop();
      }, { threshold: 0.25 }).observe(root);
    }
  }

  function boot() {
    document.querySelectorAll("[data-tree]").forEach(init);
  }
  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", boot)
    : boot();
})();
