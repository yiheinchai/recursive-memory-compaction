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

  /* The lesson as removable parts. Compression is then something you can
     watch happen — the clauses that go are struck out and then leave — rather
     than one string being swapped for another behind your back. */
  var SEGS = [
    { t: "Retry idempotent calls",                    lv: [0, 1, 2] },
    { t: " on 5xx and timeouts",                      lv: [0, 1] },
    { t: ", 3 attempts, backoff 100 / 400 / 1600 ms", lv: [0] },
    { t: ". S3 answers 200 with the error in the body, so parse the body, not the status code.", lv: [0, 1] },
    { t: "; parse bodies, not status codes.",         lv: [2] }
  ];
  var TOK = [260, 146, 92];

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
  var PROMPT_C = "same for the webhook sender";

  var OPEN_A = [
    { kind: "status", text: "Investigating… (3s · ↑ 1.4k tokens · esc to interrupt)" },
    { kind: "bullet", text: "Bash(curl -sI $BUCKET/objects/42)", out: "HTTP/1.1 200 OK" }
  ];

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
  var ACT_C = [
    { kind: "bullet", text: "Update(src/webhooks/sender.ts)", out: "Updated with 3 additions" },
    { kind: "bullet", text: "Done — first try." }
  ];

  var SPIN = ["◐", "◓", "◑", "◒"];

  function terminal(root) {
    var q = function (s) { return root.querySelector(s); };
    var stage = q(".cc-stage");
    var panes = [q("[data-pane='0']"), q("[data-pane='1']"), q("[data-pane='2']")];
    var ins = panes.map(function (p) { return p.querySelector(".pane-in"); });
    var tabs = root.querySelectorAll("[data-tab]");
    var storeSlot = q("[data-store-slot]");
    var storeNote = q("[data-store-note]");
    var typed = q("[data-typed]");
    var title = q("[data-term-title]");
    var foot = q("[data-term-foot]");
    var spin = q("[data-term-spin]");
    var rushTag = q("[data-rush-tag]");

    /* Three separate cards, because they are three separate things:
         store  — what is on disk
         ctx    — the copy injected into a session; once written it is history
         work   — the new, shorter lesson compaction produces off-thread
       Shortening the copy already sitting in the transcript would be a lie. */
    function makeCard(cls) {
      var el = h("div", cls);
      var text = h("span", "lf-text");
      var segs = SEGS.map(function (sg) {
        var sp = h("span", "seg", sg.t);
        text.appendChild(sp);
        return sp;
      });
      var tok = h("b", null, String(TOK[0]));
      var wrap = h("span", "lf-tok");
      wrap.appendChild(tok);
      wrap.appendChild(document.createTextNode(" tok"));
      el.appendChild(text);
      el.appendChild(wrap);
      return {
        el: el, segs: segs, tok: tok,
        level: function (n) {
          segs.forEach(function (sp, i) {
            sp.classList.remove("dropping");
            sp.style.display = SEGS[i].lv.indexOf(n) >= 0 ? "" : "none";
          });
          tok.textContent = TOK[n];
        }
      };
    }

    var store = makeCard("lesson-card in-store");
    var ctx = makeCard("lesson-card floating");
    var work = makeCard("lesson-card floating");
    storeSlot.appendChild(store.el);
    stage.appendChild(ctx.el);
    stage.appendChild(work.el);

    // ---- transcripts -----------------------------------------------------
    var userA = row({ kind: "user", text: PROMPT_A });
    var openA = OPEN_A.map(row);
    var grind = GRIND.concat(GRIND, GRIND).map(function (t) {
      var el = h("div", "ln grindline");
      el.appendChild(h("div", "ln-head", t));
      return el;
    });
    var closeA = CLOSE_A.map(row);
    var reflect = REFLECT.map(row);
    var gapA = h("div", "lesson-gap");
    [userA].concat(openA, grind, closeA, reflect, [gapA])
      .forEach(function (n) { ins[0].appendChild(n); });

    function session(prompt, acts) {
      var recall = row({ kind: "recall", text: "Recalling lessons…" });
      var gapTop = h("div", "lesson-gap");
      var user = row({ kind: "user", text: prompt });
      var rows = acts.map(row);
      var comp = row({ kind: "recall", text: "RMC · lesson used, work succeeded · compacting off-thread…" });
      var gapEnd = h("div", "lesson-gap");
      return {
        recall: recall, txt: recall.querySelector(".txt"),
        gapTop: gapTop, gapEnd: gapEnd, user: user, rows: rows, comp: comp,
        all: [recall, user].concat(rows, [comp]),
        nodes: [recall, gapTop, user].concat(rows, [comp, gapEnd])
      };
    }
    var B = session(PROMPT_B, ACT_B);
    var C = session(PROMPT_C, ACT_C);
    B.nodes.forEach(function (n) { ins[1].appendChild(n); });
    C.nodes.forEach(function (n) { ins[2].appendChild(n); });

    var lines = [userA].concat(openA, grind, closeA, reflect, B.all, C.all);

    // ---- primitives ------------------------------------------------------
    var timers = [];
    function later(ms, fn) { timers.push(setTimeout(fn, ms)); }
    function clearAll() { timers.forEach(clearTimeout); timers = []; }

    function roll(i, ms) {
      var over = Math.max(0, ins[i].offsetHeight - (panes[i].clientHeight - 30));
      ins[i].style.transition = ms ? "transform " + ms + "ms linear"
                                   : "transform .38s cubic-bezier(.4,0,.2,1)";
      ins[i].style.transform = "translateY(" + (-over) + "px)";
    }
    function show(n, i) { n.classList.add("on"); if (i != null) roll(i); }
    function typeInto(text, ms, done) {
      var i = 0;
      var step = function () {
        typed.textContent = text.slice(0, ++i);
        if (i < text.length) timers.push(setTimeout(step, ms));
        else if (done) timers.push(setTimeout(done, 380));
      };
      timers.push(setTimeout(step, ms));
    }
    function tab(n) {
      Array.prototype.forEach.call(tabs, function (t, i) { t.classList.toggle("on", i === n); });
      var cc = root.querySelector(".cc");
      cc.classList.toggle("on-b", n === 1);
      cc.classList.toggle("on-c", n === 2);
    }
    function at(card, target, quiet) {
      var s = stage.getBoundingClientRect(), t = target.getBoundingClientRect();
      if (quiet) card.el.style.transition = "none";
      card.el.style.width = t.width + "px";
      card.el.style.transform = "translate(" + (t.left - s.left) + "px," + (t.top - s.top) + "px)";
      if (quiet) { void card.el.offsetWidth; card.el.style.transition = ""; }
    }
    function dockInto(card, gap, paneIdx, quiet) {
      card.el.classList.add("inline");
      card.el.style.width = (panes[paneIdx].clientWidth - 36) + "px";
      gap.style.height = (card.el.offsetHeight + 10) + "px";
      roll(paneIdx);
      at(card, gap, quiet);
    }
    function countTo(card, to) {
      var from = Number(card.tok.textContent) || to, t0 = null;
      requestAnimationFrame(function tick(now) {
        if (t0 === null) t0 = now;
        var k = Math.min(1, (now - t0) / 750);
        card.tok.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
        if (k < 1) requestAnimationFrame(tick); else card.tok.textContent = TOK[to === TOK[1] ? 1 : 2];
      });
    }

    /* Strike what is going, sweep, let it leave, reflow tighter. */
    function compress(card, from, to, gap, paneIdx) {
      card.el.classList.add("compacting");
      card.segs.forEach(function (sp, i) {
        if (SEGS[i].lv.indexOf(from) >= 0 && SEGS[i].lv.indexOf(to) < 0) sp.classList.add("dropping");
      });
      later(1050, function () {
        card.level(to);
        var t0 = null, fromTok = TOK[from];
        requestAnimationFrame(function tick(now) {
          if (t0 === null) t0 = now;
          var k = Math.min(1, (now - t0) / 700);
          card.tok.textContent = Math.round(fromTok + (TOK[to] - fromTok) * (1 - Math.pow(1 - k, 3)));
          if (k < 1) requestAnimationFrame(tick); else card.tok.textContent = TOK[to];
        });
        gap.style.height = (card.el.offsetHeight + 10) + "px";
        roll(paneIdx);
        at(card, gap);
      });
      later(2000, function () { card.el.classList.remove("compacting"); });
    }

    function reset() {
      clearAll();
      lines.forEach(function (n) { n.classList.remove("on"); });
      ins.forEach(function (n) { n.style.transition = ""; n.style.transform = ""; });
      panes[0].classList.remove("dim", "rushing");
      panes.forEach(function (p) { p.classList.remove("gone"); });
      rushTag.classList.remove("on");
      [gapA, B.gapTop, B.gapEnd, C.gapTop, C.gapEnd].forEach(function (g) { g.style.height = "0px"; });
      [ctx, work].forEach(function (c) {
        c.el.className = "lesson-card floating";
        c.el.style.transform = ""; c.el.style.width = "";
        c.level(0);
      });
      store.el.className = "lesson-card in-store";
      store.level(0);
      storeSlot.classList.remove("full");
      storeNote.textContent = "empty";
      typed.textContent = ""; foot.textContent = "";
      B.txt.textContent = "Recalling lessons…";
      C.txt.textContent = "Recalling lessons…";
      title.textContent = "Fix the silent S3 retry failure";
      tab(0);
    }

    // ---- the run ---------------------------------------------------------
    function run() {
      reset();

      typeInto(PROMPT_A, 38, function () {
        typed.textContent = ""; show(userA, 0); foot.textContent = "1,180 tokens";
      });
      var t = PROMPT_A.length * 38 + 460;

      later(t + 250, function () { show(openA[0], 0); });
      later(t + 750, function () { show(openA[1], 0); });
      later(t + 1300, function () {
        grind.forEach(function (n) { n.classList.add("on"); });
        panes[0].classList.add("rushing");
        rushTag.classList.add("on");
        requestAnimationFrame(function () { roll(0, 3200); });
      });
      [1700, 2200, 2700, 3200, 3700, 4200].forEach(function (d, i) {
        later(t + d, function () {
          foot.textContent = [1600, 2200, 2700, 3300, 3800, 4200][i].toLocaleString() + " tokens";
        });
      });
      later(t + 4600, function () {
        panes[0].classList.remove("rushing"); rushTag.classList.remove("on");
      });
      later(t + 4900, function () { show(closeA[0], 0); });
      later(t + 5500, function () { show(closeA[1], 0); });
      later(t + 6000, function () { foot.textContent = "4,200 tokens to get here"; });
      later(t + 6500, function () {
        show(reflect[0], 0); title.textContent = "RMC · reflecting on session 14";
      });
      later(t + 7300, function () { show(reflect[1], 0); });
      later(t + 7900, function () {
        dockInto(work, gapA, 0, true);
        later(200, function () { work.el.classList.add("on"); });
      });
      later(t + 8900, function () {
        work.el.classList.remove("inline");
        work.el.classList.add("lift");
        panes[0].classList.add("dim");
      });
      later(t + 9800, function () {
        work.el.classList.remove("lift");
        at(work, storeSlot);
      });
      later(t + 10600, function () {
        work.el.classList.remove("on");
        store.el.classList.add("on");
        storeSlot.classList.add("full");
        storeNote.textContent = "1 lesson · L0 · 260 tok";
      });

      /* A later session: a COPY is injected, the work succeeds, and the
         compaction that follows produces a NEW card at the foot of the
         transcript. What is already in context stays exactly as it was. */
      function act(S, idx, prompt, level, startAt, titleText, doneText) {
        later(startAt, function () {
          tab(idx);
          panes.forEach(function (p, i) { p.classList.toggle("gone", i !== idx); });
          title.textContent = titleText;
          foot.textContent = "";
        });
        later(startAt + 450, function () {
          typeInto(prompt, 38, function () { typed.textContent = ""; show(S.user, idx); });
        });
        var u = startAt + 450 + prompt.length * 38 + 760;

        later(u + 150, function () { show(S.recall, idx); });
        later(u + 750, function () {
          S.txt.textContent = "RMC · 1 lesson · L" + level + " · " + TOK[level] + " tok";
          ctx.level(level);
          ctx.el.className = "lesson-card floating";
          at(ctx, storeSlot, true);           // the copy starts at the store…
          ctx.el.classList.add("on");
          requestAnimationFrame(function () { dockInto(ctx, S.gapTop, idx); });
          storeNote.textContent = "1 lesson · L" + level + " · " + TOK[level] + " tok · recalled";
        });

        var k = u + 1700;
        S.rows.forEach(function (r, i) { later(k + i * 600, function () { show(r, idx); }); });
        k += S.rows.length * 600;
        later(k + 250, function () {
          foot.textContent = (level === 0 ? "340" : "290") + " tokens · first try";
        });
        later(k + 800, function () { show(S.comp, idx); });

        // the new, shorter lesson is written at the end of the session
        later(k + 1300, function () {
          work.level(level);
          work.el.className = "lesson-card floating";
          dockInto(work, S.gapEnd, idx, true);
          later(180, function () { work.el.classList.add("on"); });
        });
        later(k + 2000, function () { compress(work, level, level + 1, S.gapEnd, idx); });
        later(k + 4300, function () {
          work.el.classList.remove("inline");
          at(work, storeSlot);
        });
        later(k + 5100, function () {
          work.el.classList.remove("on");
          store.level(level + 1);
          storeNote.textContent = "1 lesson · L" + (level + 1) + " · " + TOK[level + 1] + " tok";
        });
        later(k + 5500, function () { foot.textContent = doneText; });
        return k + 5500;
      }

      var e1 = act(B, 1, PROMPT_B, 0, t + 12000, "Add retry to the payments client",
                   "260 → 146 tok · 44% cheaper to recall");
      var e2 = act(C, 2, PROMPT_C, 1, e1 + 1600, "Same for the webhook sender",
                   "started at 260 · now 92 · same behaviour");
      later(e2 + 4200, run);
    }

    if (REDUCED) {
      reset();
      tab(2);
      panes.forEach(function (p, i) { p.classList.toggle("gone", i !== 2); });
      ctx.level(1); C.txt.textContent = "RMC · 1 lesson · L1 · 146 tok";
      C.all.forEach(function (n) { n.classList.add("on"); });
      dockInto(ctx, C.gapTop, 2, true); ctx.el.classList.add("on");
      store.level(2); store.el.classList.add("on");
      storeSlot.classList.add("full");
      storeNote.textContent = "1 lesson · L2 · 92 tok";
      title.textContent = "Same for the webhook sender";
      foot.textContent = "started at 260 · now 92 · same behaviour";
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
