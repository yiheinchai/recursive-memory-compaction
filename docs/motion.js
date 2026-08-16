/* Two animations that carry the argument, so the page does not have to.
 *
 * 1. A Claude Code window with two tabs — WITHOUT RMC and WITH RMC. The first
 *    tab plays a session the hard way: you type a prompt, and the agent grinds
 *    through dozens of exchanges before it lands. The lesson is then lifted out
 *    of that transcript, the window switches tabs by itself, and the lesson
 *    drops into the new session's context, which gets it right immediately.
 *
 * 2. One lesson card that rewrites itself shorter, twice.
 *
 * Both use reserved space. A figure that grows while it plays would shove the
 * rest of the page down on every tick, which is worse than the motion is good.
 */
(function () {
  "use strict";

  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function h(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  // ======================================================================= //
  // 1. The Claude Code window                                               //
  // ======================================================================= //

  var LESSON = "Retry idempotent calls; parse bodies, not status codes.";

  var MARK = { user: "›", status: "✱", bullet: "⏺", recall: "⋯" };

  function row(spec) {
    var el = h("div", "ln " + spec.kind);
    var head = h("div", "ln-head");
    head.appendChild(h("span", "mark", MARK[spec.kind] || ""));
    head.appendChild(h("span", "txt", spec.text));
    el.appendChild(head);
    if (spec.out) {
      var o = h("div", "ln-out");
      o.appendChild(h("span", "mark", "⎿"));
      o.appendChild(h("span", "txt", spec.out));
      el.appendChild(o);
    }
    return el;
  }

  var PROMPT_A = "the S3 retry keeps silently failing";
  var PROMPT_B = "add retry to the payments client";

  var OPEN_A = [
    { kind: "status", text: "Investigating… (3s · ↑ 1.4k tokens · esc to interrupt)" },
    { kind: "bullet", text: "Bash(curl -sI $BUCKET/objects/42)", out: "HTTP/1.1 200 OK" }
  ];

  /* The grind. Deliberately unreadable — the point is the volume, not the
     content, so it streams past blurred while a counter climbs. */
  var GRIND = [
    "⏺ Status is 200, so the client reads it as success.",
    "⏺ Update(src/storage/s3.ts)",
    "  ⎿  Updated with 6 additions and 2 removals",
    "⏺ Raising the attempt count to 5…",
    "⏺ Bash(npm test -- storage)",
    "  ⎿  1 failed, 42 passed",
    "⏺ Still failing. Adding a longer backoff…",
    "⏺ Update(src/storage/s3.ts)",
    "  ⎿  Updated with 4 additions",
    "⏺ Bash(npm test -- storage)",
    "  ⎿  1 failed, 42 passed",
    "⏺ Adding logging around the response…",
    "⏺ Read(src/storage/client.ts)",
    "  ⎿  Read 214 lines",
    "⏺ Checking whether the SDK swallows the error…",
    "⏺ Bash(rg -n 'statusCode' src/storage)",
    "  ⎿  17 matches",
    "⏺ Trying an explicit timeout instead…",
    "⏺ Update(src/storage/s3.ts)",
    "  ⎿  Updated with 9 additions and 5 removals",
    "⏺ Bash(npm test -- storage)",
    "  ⎿  1 failed, 42 passed",
    "⏺ Reverting the timeout change…",
    "⏺ Still failing."
  ];

  var CLOSE_A = [
    { kind: "user",   text: "S3 returns 200 with the error in the body" },
    { kind: "bullet", text: "Parsing the body, not the status. Fixed.", out: "42 passed" }
  ];

  var REFLECT = [
    { kind: "recall", text: "RMC · reflecting off-thread…" },
    { kind: "recall", text: "RMC · learned 1 lesson · n_7f2a · scope: global" }
  ];

  var RECALL = [
    { kind: "recall", text: "Recalling lessons…" },
    { kind: "recall", text: "RMC · 1 lesson · 146 tok" }
  ];

  var ACT_B = [
    { kind: "bullet", text: "Idempotency key set, and I'm parsing the response body." },
    { kind: "bullet", text: "Update(src/payments/client.ts)", out: "Updated with 4 additions" },
    { kind: "bullet", text: "Done — first try." }
  ];

  var SPIN = ["◐", "◓", "◑", "◒"];

  function terminal(root) {
    var q = function (s) { return root.querySelector(s); };
    var paneA = q("[data-pane='0']");
    var paneB = q("[data-pane='1']");
    var inA = paneA.querySelector(".pane-in");
    var inB = paneB.querySelector(".pane-in");
    var tabs = root.querySelectorAll("[data-tab]");
    var float = q("[data-lesson]");
    var typed = q("[data-typed]");
    var title = q("[data-term-title]");
    var foot = q("[data-term-foot]");
    var spin = q("[data-term-spin]");

    // ---- build once ------------------------------------------------------
    var userA = row({ kind: "user", text: PROMPT_A });
    var openA = OPEN_A.map(row);
    var grindBox = h("div", "grind");
    var grindInner = h("div", "grind-inner");
    GRIND.forEach(function (t) { grindInner.appendChild(h("div", "gl", t)); });
    grindBox.appendChild(grindInner);
    grindBox.appendChild(h("div", "grind-tag", "47 more exchanges"));
    var closeA = CLOSE_A.map(row);
    var reflect = REFLECT.map(row);
    // Reserved room at the foot of the transcript so the lesson has somewhere
    // to sit that is not on top of the last thing the agent said.
    var spacer = h("div", "lesson-gap");
    [userA].concat(openA, [grindBox], closeA, reflect, [spacer]).forEach(function (n) { inA.appendChild(n); });

    var recall = row(RECALL[0]);
    var slot = h("div", "lesson-slot");
    var userB = row({ kind: "user", text: PROMPT_B });
    var actB = ACT_B.map(row);
    [recall, slot].concat([userB], actB).forEach(function (n) { inB.appendChild(n); });

    float.textContent = LESSON;
    var recallTxt = recall.querySelector(".txt");

    var all = [userA].concat(openA, closeA, reflect, [recall, userB], actB);

    // ---- primitives ------------------------------------------------------
    var timers = [];
    function later(ms, fn) { timers.push(setTimeout(fn, ms)); }
    function clearAll() { timers.forEach(clearTimeout); timers = []; }

    function roll(pane, inner) {
      var room = pane.clientHeight - 30;          // pane padding, top + bottom
      var over = Math.max(0, inner.offsetHeight - room);
      inner.style.transform = "translateY(" + (-over) + "px)";
    }
    function show(n, pane, inner) {
      n.classList.add("on");
      if (pane) roll(pane, inner);
    }
    function typeInto(text, ms, done) {
      var i = 0;
      var step = function () {
        typed.textContent = text.slice(0, ++i);
        if (i < text.length) timers.push(setTimeout(step, ms));
        else if (done) timers.push(setTimeout(done, 420));
      };
      timers.push(setTimeout(step, ms));
    }
    function tab(n) {
      Array.prototype.forEach.call(tabs, function (t, i) {
        t.classList.toggle("on", i === n);
      });
      root.classList.toggle("on-b", n === 1);
    }

    function reset() {
      clearAll();
      all.forEach(function (n) { n.classList.remove("on"); });
      grindBox.classList.remove("on", "run");
      inA.style.transform = "";
      inB.style.transform = "";
      paneA.classList.remove("dim");
      spacer.style.height = "0px";
      float.style.transform = "";
      float.className = "lesson-float";
      typed.textContent = "";
      foot.textContent = "";
      title.textContent = "Fix the silent S3 retry failure";
      recallTxt.textContent = RECALL[0].text;
      tab(0);
      paneA.classList.remove("gone");
    }

    // ---- the run ---------------------------------------------------------
    function run() {
      reset();

      // Phase 1 — you type the prompt, and press enter.
      typeInto(PROMPT_A, 42, function () {
        typed.textContent = "";
        show(userA, paneA, inA);
        foot.textContent = "1,180 tokens";
      });

      var t = PROMPT_A.length * 42 + 500;

      // Phase 2 — a couple of honest steps, then the grind.
      later(t + 300, function () { show(openA[0], paneA, inA); });
      later(t + 900, function () { show(openA[1], paneA, inA); });
      later(t + 1500, function () {
        grindBox.classList.add("on");
        roll(paneA, inA);
        requestAnimationFrame(function () { grindBox.classList.add("run"); });
      });
      [1900, 2400, 2900, 3400, 3900, 4400].forEach(function (d, i) {
        later(t + d, function () { foot.textContent = [1600, 2100, 2700, 3200, 3800, 4200][i].toLocaleString() + " tokens"; });
      });

      // Phase 3 — the one exchange that mattered, crisp again.
      later(t + 4900, function () { grindBox.classList.remove("run"); });
      later(t + 5200, function () { show(closeA[0], paneA, inA); });
      later(t + 5900, function () { show(closeA[1], paneA, inA); });
      later(t + 6500, function () { foot.textContent = "4,200 tokens to get here"; });

      // Phase 4 — the session ends, and RMC reflects off the main thread.
      later(t + 7100, function () {
        show(reflect[0], paneA, inA);
        title.textContent = "RMC · reflecting on session 14";
      });
      later(t + 8100, function () { show(reflect[1], paneA, inA); });

      // Phase 5 — the lesson is lifted out of the transcript and kept.
      later(t + 8700, function () {
        var y = paneA.clientHeight - 30 - float.offsetHeight - 46;
        spacer.style.height = (float.offsetHeight + 18) + "px";
        roll(paneA, inA);
        float.style.transform = "translateY(" + y + "px)";
        float.classList.add("on");
        float.dataset.rest = y;
      });
      later(t + 9500, function () {
        float.style.transform = "translateY(" + float.dataset.rest + "px) scale(1.03)";
        float.classList.add("lift");
        paneA.classList.add("dim");
      });

      // Phase 6 — the window switches tabs by itself.
      later(t + 10500, function () {
        tab(1);
        paneA.classList.add("gone");
        title.textContent = "Add retry to the payments client";
        foot.textContent = "";
        show(recall);
      });
      later(t + 11100, function () {
        recallTxt.textContent = RECALL[1].text;
        float.style.transform = "translateY(0) scale(1)";
        float.classList.add("docked");
      });

      // Phase 7 — and this time it lands first try.
      later(t + 12100, function () {
        typeInto(PROMPT_B, 42, function () {
          typed.textContent = "";
          show(userB, paneB, inB);
        });
      });
      var u = t + 12100 + PROMPT_B.length * 42 + 500;
      later(u + 300, function () { show(actB[0], paneB, inB); });
      later(u + 900, function () { show(actB[1], paneB, inB); });
      later(u + 1500, function () { show(actB[2], paneB, inB); });
      later(u + 2000, function () { foot.textContent = "340 tokens · first try"; });

      later(u + 5200, run);
    }

    if (REDUCED) {
      reset();
      tab(1);
      paneA.classList.add("gone");
      float.classList.add("on", "docked");
      float.style.transform = "translateY(0)";
      recallTxt.textContent = RECALL[1].text;
      show(recall); show(userB); actB.forEach(show);
      title.textContent = "Add retry to the payments client";
      foot.textContent = "340 tokens · first try";
      return;
    }

    var frame = 0;
    setInterval(function () { spin.textContent = SPIN[frame++ % SPIN.length]; }, 260);

    var replay = q("[data-replay]");
    if (replay) replay.addEventListener("click", run);

    var live = false;
    new IntersectionObserver(function (e) {
      if (e[0].isIntersecting && !live) { live = true; run(); }
      else if (!e[0].isIntersecting && live) { live = false; clearAll(); }
    }, { threshold: 0.25 }).observe(root);
  }

  // ======================================================================= //
  // 2. The lesson that rewrites itself shorter                              //
  // ======================================================================= //

  var LEVELS = [
    {
      lv: "L0 · first written", tok: 260, pct: 100, drop: "",
      text: "When a remote call fails with a 5xx or a timeout, retry it — but only " +
            "if the call is idempotent. S3 answers 200 with the error inside the body, " +
            "so parse the body rather than the status code. Budget 3 attempts, with " +
            "backoff at 100 ms, 400 ms and 1.6 s."
    },
    {
      lv: "L1 · after 2 uses", tok: 195, pct: 75,
      drop: "dropped: the exact backoff timings — still held in L0",
      text: "Retry idempotent remote calls on 5xx and timeouts, 3 attempts with backoff. " +
            "S3 hides errors in a 200 body, so parse the body."
    },
    {
      lv: "L2 · after 6 uses", tok: 146, pct: 56,
      drop: "dropped: the S3 special case — still held in L1",
      text: "Retry idempotent calls; parse bodies, not status codes."
    }
  ];

  function morphCard(root) {
    var q = function (s) { return root.querySelector(s); };
    var text = q("[data-morph-text]");
    var lv = q("[data-morph-lv]");
    var tok = q("[data-morph-tok]");
    var bar = q("[data-morph-bar]");
    var drop = q("[data-morph-drop]");
    var beat = q("[data-morph-beat]");
    var dots = root.querySelectorAll("[data-morph-step]");

    var i = 0;
    var shown = LEVELS[0].tok;   // the number on screen, tracked here rather
    var raf = null;              // than read back out of the DOM

    function count(to) {
      if (REDUCED) { shown = to; tok.textContent = to; return; }
      var from = shown, t0 = null;
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(function tick(now) {
        if (t0 === null) t0 = now;
        var k = Math.min(1, Math.max(0, (now - t0) / 520));
        shown = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
        tok.textContent = shown;
        if (k < 1) raf = requestAnimationFrame(tick);
        else { shown = to; tok.textContent = to; }
      });
    }

    function go(n) {
      i = ((n % LEVELS.length) + LEVELS.length) % LEVELS.length;
      var s = LEVELS[i];
      lv.textContent = s.lv;
      bar.style.width = s.pct + "%";
      count(s.tok);

      text.style.opacity = "0";
      setTimeout(function () {
        text.textContent = s.text;
        text.style.opacity = "1";
      }, REDUCED ? 0 : 190);

      drop.textContent = s.drop;
      drop.classList.toggle("show", !!s.drop);
      if (beat) beat.classList.toggle("show", i > 0);
      Array.prototype.forEach.call(dots, function (d, k) {
        d.setAttribute("aria-selected", String(k === i));
      });
    }

    Array.prototype.forEach.call(dots, function (d, k) {
      d.addEventListener("click", function () { stop(); go(k); });
    });

    var timer = null;
    function play() { stop(); timer = setInterval(function () { go(i + 1); }, 3600); }
    function stop() { clearInterval(timer); timer = null; }

    go(0);
    if (REDUCED) return;
    root.addEventListener("mouseenter", stop);
    root.addEventListener("mouseleave", play);
    new IntersectionObserver(function (e) {
      e[0].isIntersecting ? play() : stop();
    }, { threshold: 0.3 }).observe(root);
  }

  function boot() {
    document.querySelectorAll("[data-term]").forEach(terminal);
    document.querySelectorAll("[data-morph]").forEach(morphCard);
  }
  document.readyState === "loading"
    ? document.addEventListener("DOMContentLoaded", boot)
    : boot();
})();
