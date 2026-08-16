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

  var LESSON_LONG  = "Retry idempotent remote calls on 5xx and timeouts. S3 answers 200 " +
                     "with the error in the body — parse the body, not the status. " +
                     "3 attempts; backoff 100 / 400 / 1600 ms.";
  var LESSON_SHORT = "Retry idempotent calls; parse bodies, not status codes.";

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

  /* The grind. These are ordinary transcript rows, not a widget — the whole
     terminal reels through them, because that is what a long session looks
     like. Blurred just enough that you read the motion, not the words. */
  var GRIND = [
    "⏺ Status is 200, so the client reads it as success.",
    "⏺ Update(src/storage/s3.ts)",
    "  ⎿  Updated with 6 additions and 2 removals",
    "⏺ Raising the attempt count to 5…",
    "⏺ Bash(npm test -- storage)",
    "  ⎿  1 failed, 42 passed",
    "⏺ Still failing. Adding a longer backoff…",
    "⏺ Update(src/storage/retry.ts)",
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
    "⏺ Inspecting the raw response headers…",
    "⏺ Bash(curl -s $BUCKET/objects/42 | head -c 200)",
    "  ⎿  <?xml version=\"1.0\"?><Error><Code>SlowDown</Code>",
    "⏺ Adding a jittered backoff…",
    "⏺ Update(src/storage/retry.ts)",
    "  ⎿  Updated with 11 additions",
    "⏺ Bash(npm test -- storage)",
    "  ⎿  1 failed, 42 passed",
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

  var ACT_B = [
    { kind: "bullet", text: "Idempotency key set, and I'm parsing the response body." },
    { kind: "bullet", text: "Update(src/payments/client.ts)", out: "Updated with 4 additions" },
    { kind: "bullet", text: "Done — first try." }
  ];

  var COMPACT = { kind: "recall", text: "RMC · lesson used, work succeeded · compacting…" };

  var SPIN = ["◐", "◓", "◑", "◒"];

  function terminal(root) {
    var q = function (s) { return root.querySelector(s); };
    var stage = q(".cc-stage");
    var paneA = q("[data-pane='0']");
    var paneB = q("[data-pane='1']");
    var inA = paneA.querySelector(".pane-in");
    var inB = paneB.querySelector(".pane-in");
    var tabs = root.querySelectorAll("[data-tab]");
    var float = q("[data-lesson]");
    var lText = q("[data-lesson-text]");
    var lTok = q("[data-lesson-tok]");
    var storeSlot = q("[data-store-slot]");
    var storeNote = q("[data-store-note]");
    var typed = q("[data-typed]");
    var title = q("[data-term-title]");
    var foot = q("[data-term-foot]");
    var spin = q("[data-term-spin]");
    var rushTag = q("[data-rush-tag]");

    // ---- build once ------------------------------------------------------
    var userA = row({ kind: "user", text: PROMPT_A });
    var openA = OPEN_A.map(row);
    var grind = GRIND.map(function (t) {
      var el = h("div", "ln grindline");
      el.appendChild(h("div", "ln-head", t));
      return el;
    });
    var closeA = CLOSE_A.map(row);
    var reflect = REFLECT.map(row);
    var gapA = h("div", "lesson-gap");
    [userA].concat(openA, grind, closeA, reflect, [gapA])
      .forEach(function (n) { inA.appendChild(n); });

    var recall = row({ kind: "recall", text: "Recalling lessons…" });
    var slotB = h("div", "lesson-gap");
    var userB = row({ kind: "user", text: PROMPT_B });
    var actB = ACT_B.map(row);
    var compactLine = row(COMPACT);
    [recall, slotB, userB].concat(actB, [compactLine])
      .forEach(function (n) { inB.appendChild(n); });

    var recallTxt = recall.querySelector(".txt");
    var lines = [userA].concat(openA, grind, closeA, reflect,
                               [recall, userB], actB, [compactLine]);

    // ---- primitives ------------------------------------------------------
    var timers = [];
    function later(ms, fn) { timers.push(setTimeout(fn, ms)); }
    function clearAll() { timers.forEach(clearTimeout); timers = []; }

    function roll(pane, inner, ms) {
      var over = Math.max(0, inner.offsetHeight - (pane.clientHeight - 30));
      inner.style.transition = ms
        ? "transform " + ms + "ms linear"
        : "transform .38s cubic-bezier(.4,0,.2,1)";
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
      Array.prototype.forEach.call(tabs, function (t, i) { t.classList.toggle("on", i === n); });
      root.querySelector(".cc").classList.toggle("on-b", n === 1);
    }

    /* Fly the lesson card to sit exactly over a target box. Every target is
       inset the same amount from the stage, so no width tween is needed. */
    function flyTo(target) {
      var s = stage.getBoundingClientRect();
      var t = target.getBoundingClientRect();
      float.style.width = t.width + "px";
      float.style.transform =
        "translate(" + (t.left - s.left) + "px," + (t.top - s.top) + "px)";
    }

    function reset() {
      clearAll();
      lines.forEach(function (n) { n.classList.remove("on"); });
      inA.style.transition = ""; inA.style.transform = "";
      inB.style.transition = ""; inB.style.transform = "";
      paneA.classList.remove("dim", "rushing");
      rushTag.classList.remove("on");
      gapA.style.height = "0px";
      float.className = "lesson-float";
      float.style.transform = "";
      lText.textContent = LESSON_LONG;
      lTok.textContent = "260";
      storeSlot.classList.remove("full");
      storeNote.textContent = "empty";
      typed.textContent = "";
      foot.textContent = "";
      recallTxt.textContent = "Recalling lessons…";
      title.textContent = "Fix the silent S3 retry failure";
      tab(0);
      paneA.classList.remove("gone");
    }

    // ---- the run ---------------------------------------------------------
    function run() {
      reset();

      typeInto(PROMPT_A, 42, function () {
        typed.textContent = "";
        show(userA, paneA, inA);
        foot.textContent = "1,180 tokens";
      });
      var t = PROMPT_A.length * 42 + 500;

      later(t + 250, function () { show(openA[0], paneA, inA); });
      later(t + 800, function () { show(openA[1], paneA, inA); });

      // The rush: reveal the whole grind at once, then reel the transcript
      // through it at a constant speed. The terminal scrolls, not a widget.
      later(t + 1400, function () {
        grind.forEach(function (n) { n.classList.add("on"); });
        paneA.classList.add("rushing");
        rushTag.classList.add("on");
        requestAnimationFrame(function () { roll(paneA, inA, 3600); });
      });
      [1800, 2400, 3000, 3600, 4200, 4800].forEach(function (d, i) {
        later(t + d, function () {
          foot.textContent = [1600, 2200, 2700, 3300, 3800, 4200][i].toLocaleString() + " tokens";
        });
      });
      later(t + 5100, function () {
        paneA.classList.remove("rushing");
        rushTag.classList.remove("on");
      });

      later(t + 5500, function () { show(closeA[0], paneA, inA); });
      later(t + 6200, function () { show(closeA[1], paneA, inA); });
      later(t + 6800, function () { foot.textContent = "4,200 tokens to get here"; });

      // Reflection, exactly as RMC announces it.
      later(t + 7300, function () {
        show(reflect[0], paneA, inA);
        title.textContent = "RMC · reflecting on session 14";
      });
      later(t + 8200, function () { show(reflect[1], paneA, inA); });

      // The lesson is written, then filed in the store.
      later(t + 8900, function () {
        gapA.style.height = "62px";
        roll(paneA, inA);
        later(340, function () {
          flyTo(gapA);
          float.classList.add("on");
        });
      });
      later(t + 10000, function () { float.classList.add("lift"); paneA.classList.add("dim"); });
      later(t + 10900, function () {
        float.classList.remove("lift");
        flyTo(storeSlot);
        storeSlot.classList.add("full");
        storeNote.textContent = "1 lesson · 260 tok";
      });

      // Session two.
      later(t + 12300, function () {
        tab(1);
        paneA.classList.add("gone");
        title.textContent = "Add retry to the payments client";
        foot.textContent = "";
      });
      later(t + 12900, function () {
        typeInto(PROMPT_B, 42, function () {
          typed.textContent = "";
          show(userB, paneB, inB);
        });
      });
      var u = t + 12900 + PROMPT_B.length * 42 + 920;

      later(u + 200, function () { show(recall, paneB, inB); });
      later(u + 900, function () {
        recallTxt.textContent = "RMC · 1 lesson · 260 tok";
        storeSlot.classList.remove("full");
        storeNote.textContent = "recalled";
        flyTo(slotB);
      });
      later(u + 1800, function () { show(actB[0], paneB, inB); });
      later(u + 2400, function () { show(actB[1], paneB, inB); });
      later(u + 3100, function () { show(actB[2], paneB, inB); });
      later(u + 3600, function () { foot.textContent = "340 tokens · first try"; });

      // Used, and it worked — so it earns a shorter form on the way back.
      later(u + 4300, function () { show(compactLine, paneB, inB); });
      later(u + 5100, function () {
        float.classList.add("compacting");
        lText.textContent = LESSON_SHORT;
        lTok.textContent = "146";
        later(60, function () { flyTo(slotB); });
      });
      later(u + 6100, function () {
        float.classList.remove("compacting");
        flyTo(storeSlot);
        storeSlot.classList.add("full");
        storeNote.textContent = "1 lesson · 146 tok";
      });
      later(u + 6900, function () { foot.textContent = "next recall costs 44% less"; });

      later(u + 10500, run);
    }

    if (REDUCED) {
      reset();
      tab(1);
      paneA.classList.add("gone");
      lText.textContent = LESSON_SHORT; lTok.textContent = "146";
      recallTxt.textContent = "RMC · 1 lesson · 146 tok";
      [recall, userB].concat(actB).forEach(function (n) { n.classList.add("on"); });
      storeSlot.classList.add("full");
      storeNote.textContent = "1 lesson · 146 tok";
      float.classList.add("on");
      flyTo(storeSlot);
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
    }, { threshold: 0.2 }).observe(root);
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
