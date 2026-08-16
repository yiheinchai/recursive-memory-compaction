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
    { t: "; parse bodies, not status codes.",         lv: [2] },
    // level 3 is the shared parent the merge produces
    { t: "Key every outbound call for idempotency before retrying it, and judge success from the body, not the status.", lv: [3] }
  ];
  var TOK = [260, 146, 92, 118];

  /* A second lesson, learned in a different repo. It never compresses on its
     own here — its role is to be needed alongside the first one, which is what
     earns the pair a shared parent. */
  var SEGS_B = [{ t: "Give every outbound request an idempotency key before you retry it.", lv: [0] }];
  var TOK_B = [92];

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
    var q = function (sel) { return root.querySelector(sel); };
    var stage = q(".cc-stage");
    var panes = [q("[data-pane='0']"), q("[data-pane='1']"), q("[data-pane='2']")];
    var ins = panes.map(function (p) { return p.querySelector(".pane-in"); });
    var panY = [0, 0, 0];
    var tabs = root.querySelectorAll("[data-tab]");
    var slots = root.querySelectorAll("[data-store-slot]");
    var storeNote = q("[data-store-note]");
    var typed = q("[data-typed]");
    var title = q("[data-term-title]");
    var foot = q("[data-term-foot]");
    var spin = q("[data-term-spin]");
    var rushTag = q("[data-rush-tag]");

    /* Cards are REAL children of wherever they live — a line in the transcript
       or a row in the store. Nothing is absolutely positioned over the
       terminal, so nothing can drift out of sync when the transcript scrolls.
       Movement is animated by flying a throwaway ghost along the path while
       the real element sits, hidden, at its destination. */
    function makeCard(SPEC, TOKS) {
      SPEC = SPEC || SEGS; TOKS = TOKS || TOK;
      var el = h("div", "lesson-card");
      var text = h("span", "lf-text");
      var segs = SPEC.map(function (sg) {
        var sp = h("span", "seg", sg.t);
        text.appendChild(sp);
        return sp;
      });
      var tok = h("b", null, String(TOKS[0]));
      var wrap = h("span", "lf-tok");
      wrap.appendChild(tok);
      wrap.appendChild(document.createTextNode(" tok"));
      el.appendChild(text); el.appendChild(wrap);
      return {
        el: el, segs: segs, tok: tok, spec: SPEC, toks: TOKS,
        level: function (n) {
          segs.forEach(function (sp, i) {
            sp.classList.remove("dropping");
            sp.style.display = SPEC[i].lv.indexOf(n) >= 0 ? "" : "none";
          });
          tok.textContent = TOKS[n];
        }
      };
    }

    var limbo = h("div", "limbo");
    stage.appendChild(limbo);
    var ctx, ctxB, work, workB;
    function freshCards() {
      while (limbo.firstChild) limbo.removeChild(limbo.firstChild);
      ctx = makeCard(); ctxB = makeCard(SEGS_B, TOK_B);
      work = makeCard(); workB = makeCard(SEGS_B, TOK_B);
      [ctx, ctxB, work, workB].forEach(function (c) { limbo.appendChild(c.el); });
    }
    freshCards();

    var tiers = root.querySelectorAll("[data-store-slot]");   // [L2, L1, L0]
    function tierFor(level) { return tiers[2 - level]; }

    /* Park a card in the store tree as a permanent node, and mark whichever
       level is now the apex — the only row recall ever reads. */
    function fileAt(card, level) {
      travel(card, tierFor(level), "in-store stored");
      later(820, function () { markApex(level); });   // after it lands, so the fade shows
    }
    function markApex(level) {
      Array.prototype.forEach.call(root.querySelectorAll(".tier"), function (t) {
        var lv = Number(t.getAttribute("data-tier"));
        t.classList.toggle("live", t.querySelector(".lesson-card") !== null);
        t.classList.toggle("apex", lv === level);
      });
    }

    // ---- transcripts -----------------------------------------------------
    function mount() { return h("div", "lesson-mount"); }

    var userA = row({ kind: "user", text: PROMPT_A });
    var openA = OPEN_A.map(row);
    var grind = GRIND.concat(GRIND, GRIND).map(function (t) {
      var el = h("div", "ln grindline");
      el.appendChild(h("div", "ln-head", t));
      return el;
    });
    var closeA = CLOSE_A.map(row);
    var reflect = REFLECT.map(row);
    var mountA = mount();
    [userA].concat(openA, grind, closeA, reflect, [mountA])
      .forEach(function (n) { ins[0].appendChild(n); });

    function session(prompt, acts, isPair) {
      var recall = row({ kind: "recall", text: "Recalling lessons…" });
      var top = mount(), end = mount();
      var user = row({ kind: "user", text: prompt });
      var rows = acts.map(row);
      var comp = row({ kind: "recall", text: isPair
        ? "RMC · both lessons used together, 3rd time · merging into a shared parent…"
        : "RMC · lesson used, work succeeded · compacting off-thread…" });
      return {
        recall: recall, txt: recall.querySelector(".txt"),
        top: top, end: end, user: user, rows: rows, comp: comp,
        all: [recall, user].concat(rows, [comp]),
        nodes: [recall, top, user].concat(rows, [comp, end])
      };
    }
    var B = session(PROMPT_B, ACT_B, false);
    var C = session(PROMPT_C, ACT_C, true);
    B.nodes.forEach(function (n) { ins[1].appendChild(n); });
    C.nodes.forEach(function (n) { ins[2].appendChild(n); });

    var lines = [userA].concat(openA, grind, closeA, reflect, B.all, C.all);

    // ---- primitives ------------------------------------------------------
    /* The timeline is a plan of absolute-time steps rather than a pile of
       timeouts, so it can be paused and scrubbed: seeking is just "reset, then
       replay every step up to T with transitions switched off". */
    var plan = [], timers = [], seeking = false;
    function at(ms, fn) { plan.push({ at: ms, fn: fn }); }
    function later(ms, fn) {              // nested, cosmetic, relative delays
      if (seeking) { fn(); return; }
      timers.push(setTimeout(fn, ms));
    }
    function clearAll() { timers.forEach(clearTimeout); timers = []; }

    function roll(i, ms) {
      if (seeking) ms = 0;
      var over = Math.max(0, ins[i].offsetHeight - (panes[i].clientHeight - 30));
      panY[i] = -over;
      ins[i].style.transition = ms ? "transform " + ms + "ms linear"
                                   : "transform .38s cubic-bezier(.4,0,.2,1)";
      ins[i].style.transform = "translateY(" + panY[i] + "px)";
    }
    function show(n, i) { n.classList.add("on"); if (i != null) roll(i); }
    function typeInto(text, ms, done) {
      if (seeking) { typed.textContent = ""; if (done) done(); return; }
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

    /* Move a card for real, then fly a ghost along the path it took. The pane
       scroll is applied first and its delta folded into the target, or the
       ghost would land where the transcript used to be. */
    function travel(card, dest, cls, paneIdx, originEl) {
      var el = card.el;
      // An explicit origin lets a copy launch from the node it was copied
      // from, rather than from wherever the copy happened to be parked.
      var first = (originEl || el).getBoundingClientRect();
      var wasVisible = first.width > 0 &&
        (originEl ? originEl.offsetParent !== null : el.offsetParent !== null);
      var prevY = paneIdx != null ? panY[paneIdx] : 0;

      el.className = "lesson-card " + (cls || "");
      dest.appendChild(el);
      if (paneIdx != null) roll(paneIdx);

      var last = el.getBoundingClientRect();
      var shiftY = paneIdx != null ? (panY[paneIdx] - prevY) : 0;
      if (seeking) return;
      if (!wasVisible) {
        el.classList.add("pop");
        later(400, function () { el.classList.remove("pop"); });
        return;
      }

      var sr = stage.getBoundingClientRect();
      var ghost = el.cloneNode(true);
      ghost.className = "lesson-card ghost";       // one look for the whole flight
      ghost.style.left = (first.left - sr.left) + "px";
      ghost.style.top = (first.top - sr.top) + "px";
      ghost.style.width = first.width + "px";
      stage.appendChild(ghost);
      el.style.visibility = "hidden";
      void ghost.offsetWidth;
      ghost.style.width = last.width + "px";
      ghost.style.transform = "translate(" + (last.left - first.left) + "px," +
                              (last.top + shiftY - first.top) + "px)";
      later(760, function () {
        if (ghost.parentNode) ghost.parentNode.removeChild(ghost);
        el.style.visibility = "";
      });
    }

    function countCard(card, from, to, ms) {
      if (seeking) { card.tok.textContent = to; return; }
      var t0 = null;
      requestAnimationFrame(function tick(now) {
        if (t0 === null) t0 = now;
        var k = Math.min(1, (now - t0) / (ms || 750));
        card.tok.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
        if (k < 1) requestAnimationFrame(tick); else card.tok.textContent = to;
      });
    }

    /* Strike what is going, sweep, let it leave, reflow tighter. */
    function compress(card, from, to, paneIdx) {
      card.el.classList.add("compacting");
      card.segs.forEach(function (sp, i) {
        if (card.spec[i].lv.indexOf(from) >= 0 && card.spec[i].lv.indexOf(to) < 0) {
          sp.classList.add("dropping");
        }
      });
      later(1050, function () {
        var was = Number(card.tok.textContent);
        card.level(to);
        countCard(card, was, TOK[to], 700);
        roll(paneIdx);
      });
      later(2000, function () { card.el.classList.remove("compacting"); });
    }

    function reset() {
      clearAll();
      Array.prototype.forEach.call(stage.querySelectorAll(".ghost"), function (g) {
        g.parentNode.removeChild(g);
      });
      Array.prototype.forEach.call(root.querySelectorAll(".merge-row"), function (r) {
        r.parentNode.removeChild(r);
      });
      lines.forEach(function (n) { n.classList.remove("on"); });
      ins.forEach(function (n, i) { n.style.transition = ""; n.style.transform = ""; panY[i] = 0; });
      panes[0].classList.remove("dim", "rushing");
      panes.forEach(function (p) { p.classList.remove("gone"); });
      rushTag.classList.remove("on");
      // Every place a card can come to rest must be emptied, or a replay
      // stacks another copy on top of the last one.
      Array.prototype.forEach.call(root.querySelectorAll(".lesson-mount"), function (m) {
        while (m.firstChild) m.removeChild(m.firstChild);
        m.style.height = "";
      });
      Array.prototype.forEach.call(tiers, function (sl) {
        while (sl.firstChild) sl.removeChild(sl.firstChild);
      });
      freshCards();
      Array.prototype.forEach.call(root.querySelectorAll(".tier"), function (t) {
        t.classList.remove("live", "apex");
      });
      typed.textContent = ""; foot.textContent = "";
      B.txt.textContent = "Recalling lessons…";
      C.txt.textContent = "Recalling lessons…";
      storeNote.textContent = "empty";
      title.textContent = "Fix the silent S3 retry failure";
      tab(0);
    }

    // ---- the plan --------------------------------------------------------
    function build() {
      at(0, function () {
        typeInto(PROMPT_A, 38, function () {
          typed.textContent = ""; show(userA, 0); foot.textContent = "1,180 tokens";
        });
      });
      var t = PROMPT_A.length * 38 + 460;

      at(t + 250, function () { show(openA[0], 0); });
      at(t + 750, function () { show(openA[1], 0); });
      at(t + 1300, function () {
        grind.forEach(function (n) { n.classList.add("on"); });
        panes[0].classList.add("rushing");
        rushTag.classList.add("on");
        requestAnimationFrame(function () { roll(0, 1850); });
      });
      [1500, 1780, 2060, 2340, 2620, 2900].forEach(function (d, i) {
        at(t + d, function () {
          foot.textContent = [1600, 2200, 2700, 3300, 3800, 4200][i].toLocaleString() + " tokens";
        });
      });
      at(t + 3250, function () {
        panes[0].classList.remove("rushing"); rushTag.classList.remove("on");
      });
      at(t + 3550, function () { show(closeA[0], 0); });
      at(t + 4150, function () { show(closeA[1], 0); });
      at(t + 4650, function () { foot.textContent = "4,200 tokens to get here"; });
      at(t + 5150, function () {
        show(reflect[0], 0); title.textContent = "RMC · reflecting on session 14";
      });
      at(t + 5900, function () { show(reflect[1], 0); });
      at(t + 6550, function () { work.level(0); travel(work, mountA, "inline", 0); });
      at(t + 7500, function () { work.el.classList.add("lift"); panes[0].classList.add("dim"); });
      at(t + 8350, function () {
        work.el.classList.remove("lift");
        fileAt(work, 0);
        storeNote.textContent = "1 lesson · apex L0 · 260 tok";
        work = makeCard(); limbo.appendChild(work.el);   // next session needs a fresh one
      });

      function act(S, idx, prompt, level, startAt, titleText, doneText, pair) {
        at(startAt, function () {
          tab(idx);
          panes.forEach(function (p, i) { p.classList.toggle("gone", i !== idx); });
          title.textContent = titleText;
          foot.textContent = "";
        });
        at(startAt + 450, function () {
          typeInto(prompt, 38, function () { typed.textContent = ""; show(S.user, idx); });
        });
        var u = startAt + 450 + prompt.length * 38 + 760;

        at(u + 150, function () { show(S.recall, idx); });
        at(u + 750, function () {
          var total = pair ? TOK[level] + TOK_B[0] : TOK[level];
          S.txt.textContent = pair
            ? "RMC · 2 lessons · " + total + " tok"
            : "RMC · 1 lesson · L" + level + " · " + TOK[level] + " tok";
          var nodes = tierFor(level).children;
          ctx.level(level);
          travel(ctx, S.top, "inline", idx, nodes[0]);
          if (pair) {
            ctxB.level(0);
            travel(ctxB, S.top, "inline", idx, nodes[1] || nodes[0]);
          }
          storeNote.textContent = (pair ? "2 lessons · " : "1 lesson · L" + level + " · ") +
                                  total + " tok · recalled";
        });

        var k = u + 1700;
        S.rows.forEach(function (r, i) { at(k + i * 600, function () { show(r, idx); }); });
        k += S.rows.length * 600;
        at(k + 250, function () {
          foot.textContent = (level === 0 ? "340" : "290") + " tokens · first try";
        });
        at(k + 800, function () { show(S.comp, idx); });

        if (!pair) {
          at(k + 1300, function () { work.level(level); travel(work, S.end, "inline", idx); });
          at(k + 2000, function () { compress(work, level, level + 1, idx); });
          at(k + 4200, function () {
            fileAt(work, level + 1);
            storeNote.textContent = "1 lesson · apex L" + (level + 1) + " · " + TOK[level + 1] + " tok";
            work = makeCard(); limbo.appendChild(work.el);
          });
          at(k + 5400, function () { foot.textContent = doneText; });
          return k + 5400;
        }

        /* Both were used, so they earn a shared parent. The two lessons are
           written side by side at the foot of the transcript, slide into each
           other, and what comes out is one new lesson. */
        var row2 = h("div", "merge-row");
        var burst = h("span", "burst");
        at(k + 1300, function () {
          work.level(level); workB.level(0);
          row2.appendChild(burst);
          S.end.appendChild(row2);
          travel(work, row2, "inline merge-a", idx);
          travel(workB, row2, "inline merge-b", idx);
        });
        at(k + 2200, function () { row2.classList.add("armed"); });
        at(k + 3050, function () { row2.classList.add("clash"); });
        at(k + 3850, function () {
          var was = TOK[level] + TOK_B[0];
          row2.classList.add("boom");
          workB.el.className = "lesson-card";
          limbo.appendChild(workB.el);
          row2.classList.remove("clash", "armed");
          work.el.className = "lesson-card inline merge-a born";
          work.level(3);
          countCard(work, was, TOK[3], 850);
          roll(idx);
        });
        at(k + 4800, function () {
          row2.classList.remove("boom");
          work.el.classList.remove("born");
        });
        at(k + 5300, function () {
          fileAt(work, 2);
          storeNote.textContent = "3 nodes · apex L2 · shared parent · " + TOK[3] + " tok";
          work = makeCard(); limbo.appendChild(work.el);
        });
        at(k + 6500, function () { foot.textContent = doneText; });
        return k + 6500;
      }

      var e1 = act(B, 1, PROMPT_B, 0, t + 10600, "Add retry to the payments client",
                   "260 → 146 tok · 44% cheaper to recall", false);

      // Something learned in a different repo lands in the same global store.
      at(e1 + 700, function () {
        var b = makeCard(SEGS_B, TOK_B);
        limbo.appendChild(b.el);
        b.level(0);
        fileAt(b, 1);
        tierFor(1).classList.add("landed");
        later(1000, function () { tierFor(1).classList.remove("landed"); });
        storeNote.textContent = "2 lessons · +1 learned in another repo";
      });
      var e2 = act(C, 2, PROMPT_C, 1, e1 + 2400, "Same for the webhook sender",
                   "2 lessons · 238 tok  →  one shared parent · 118 tok", true);
      return e2 + 4000;
    }

    // ---- transport: play, pause, scrub -----------------------------------
    var DUR = 0, pos = 0, playing = false, fired = 0, wall = 0, raf = null;
    var pp = q("[data-playpause]"), scrub = q("[data-scrub]"), clock = q("[data-clock]");

    function fmt(ms) {
      var sec = Math.max(0, Math.round(ms / 1000));
      return Math.floor(sec / 60) + ":" + ("0" + (sec % 60)).slice(-2);
    }
    function paintBar() {
      if (scrub && document.activeElement !== scrub) {
        scrub.value = String(Math.round((pos / DUR) * 1000));
      }
      if (clock) clock.textContent = fmt(pos) + " / " + fmt(DUR);
    }
    /* Seeking is a replay, not a rewind: reset, then run every step up to T
       with transitions suppressed so the frame lands instantly. */
    function seek(ms) {
      seeking = true;
      root.classList.add("seeking");
      reset();
      fired = 0;
      while (fired < plan.length && plan[fired].at <= ms) { plan[fired].fn(); fired++; }
      pos = ms;
      wall = performance.now() - ms;
      void root.offsetWidth;
      seeking = false;
      root.classList.remove("seeking");
      paintBar();
    }
    function tick(now) {
      if (!playing) return;
      pos = now - wall;
      while (fired < plan.length && plan[fired].at <= pos) { plan[fired].fn(); fired++; }
      if (pos >= DUR) { seek(0); wall = performance.now(); }
      paintBar();
      raf = requestAnimationFrame(tick);
    }
    function play() {
      if (playing) return;
      playing = true;
      wall = performance.now() - pos;
      if (pp) { pp.textContent = "❚❚"; pp.setAttribute("aria-label", "Pause"); }
      raf = requestAnimationFrame(tick);
    }
    function pause() {
      if (!playing) return;
      playing = false;
      cancelAnimationFrame(raf);
      clearAll();
      if (pp) { pp.textContent = "▶"; pp.setAttribute("aria-label", "Play"); }
    }

    plan = [];
    DUR = build();
    seek(0);

    if (REDUCED) {
      seek(DUR - 500);
      if (pp) pp.parentNode.removeChild(pp);
      return;
    }

    var frame = 0;
    setInterval(function () { spin.textContent = SPIN[frame++ % SPIN.length]; }, 260);

    if (pp) pp.addEventListener("click", function () { playing ? pause() : play(); });
    if (scrub) {
      scrub.addEventListener("input", function () {
        pause();
        seek((Number(scrub.value) / 1000) * DUR);
      });
    }
    var replay = q("[data-replay]");
    if (replay) replay.addEventListener("click", function () { seek(0); play(); });

    // Only run while it is on screen; remember where the viewer left it.
    new IntersectionObserver(function (e) {
      e[0].isIntersecting ? play() : pause();
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
