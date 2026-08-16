/* Two animations that carry the argument, so the page does not have to.
 *
 * 1. A Claude Code window that plays one session, distils it to a single line,
 *    and then starts a NEW session with that line already in context. The
 *    chrome is copied from the real thing on purpose: nobody should need to be
 *    told they are looking at a Claude Code session.
 *
 * 2. One lesson card that rewrites itself shorter, twice.
 *
 * Both use reserved slots. A figure that grows while it plays would shove the
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

  /* Animate a box between its current and its next natural height. */
  function morph(box, mutate, ms) {
    var h0 = box.offsetHeight;
    mutate();
    box.style.height = "auto";
    var h1 = box.offsetHeight;
    if (REDUCED) { box.style.height = ""; return; }
    box.style.height = h0 + "px";
    void box.offsetHeight;
    box.style.transition = "height " + ms + "ms cubic-bezier(.4,0,.2,1)";
    box.style.height = h1 + "px";
    setTimeout(function () {
      box.style.transition = "";
      box.style.height = "";
    }, ms + 50);
  }

  function Timeline(steps, loopAt) {
    var handles = [];
    return {
      stop: function () { handles.forEach(clearTimeout); handles = []; },
      play: function () {
        this.stop();
        var self = this;
        steps.forEach(function (s) {
          handles.push(setTimeout(s.run, REDUCED ? 0 : s.at));
        });
        if (loopAt && !REDUCED) {
          handles.push(setTimeout(function () { self.play(); }, loopAt));
        }
      }
    };
  }

  // ======================================================================= //
  // 1. The Claude Code window                                               //
  // ======================================================================= //

  var LESSON = "Retry idempotent calls; parse bodies, not status codes.";

  //  kind:  user | status | bullet | recall | lesson
  //  out:   the ⎿ line underneath a tool call
  var ACT1 = [
    { kind: "user",   text: "the S3 retry keeps silently failing" },
    { kind: "status", text: "Investigating… (3s · ↑ 1.4k tokens · esc to interrupt)" },
    { kind: "bullet", text: "Bash(curl -sI $BUCKET/objects/42)", out: "HTTP/1.1 200 OK" },
    { kind: "bullet", text: "Status is 200, so the client reads it as success. Raising the attempt count." },
    { kind: "bullet", text: "Update(src/storage/s3.ts)", out: "Updated with 6 additions and 2 removals" },
    { kind: "bullet", text: "Still failing. Adding logging around the response…" },
    { kind: "user",   text: "S3 returns 200 with the error in the body" },
    { kind: "bullet", text: "Parsing the body, not the status. Fixed." }
  ];

  var ACT3 = [
    { kind: "user",   text: "add retry to the payments client" },
    { kind: "bullet", text: "Idempotency key set, and I'm parsing the response body." },
    { kind: "bullet", text: "Update(src/payments/client.ts)", out: "Updated with 4 additions" },
    { kind: "bullet", text: "Done — first try." }
  ];

  var MARK = { user: "›", status: "✱", bullet: "⏺", recall: "⋯", lesson: "" };

  function row(spec) {
    var el = h("div", "ln " + spec.kind);
    var head = h("div", "ln-head");
    head.appendChild(h("span", "mark", MARK[spec.kind]));
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

  var SPIN = ["◐", "◓", "◑", "◒"];

  function terminal(root) {
    var body = root.querySelector("[data-term-body]");
    var title = root.querySelector("[data-term-title]");
    var foot = root.querySelector("[data-term-foot]");
    var spin = root.querySelector("[data-term-spin]");

    var a1 = ACT1.map(row);
    var recall = row({ kind: "recall", text: "RMC · 1 lesson recalled · 146 tok" });
    var lesson = row({ kind: "lesson", text: LESSON });
    var a3 = ACT3.map(row);
    a1.concat([recall, lesson], a3).forEach(function (n) { body.appendChild(n); });

    function show(n) { n.classList.remove("leaving"); n.classList.add("on"); }
    function hide(n) {
      if (!n.classList.contains("on")) return;
      n.classList.add("leaving");
      setTimeout(function () { n.classList.remove("on", "leaving"); }, 320);
    }
    function reset() {
      a1.concat([recall, lesson], a3).forEach(function (n) {
        n.classList.remove("on", "leaving");
      });
      title.textContent = "Fix the silent S3 retry failure";
      foot.textContent = "";
    }

    var steps = [];
    ACT1.forEach(function (_, i) {
      steps.push({ at: 600 + i * 480, run: function () { show(a1[i]); } });
    });
    steps.push({ at: 4500, run: function () { foot.textContent = "4,200 tokens to get here"; } });
    steps.push({ at: 5500, run: function () { title.textContent = "RMC · reflecting on session 14"; } });
    steps.push({ at: 6300, run: function () {
      a1.forEach(hide);
      setTimeout(function () { show(lesson); }, 280);
    } });
    steps.push({ at: 7000, run: function () {
      foot.textContent = "146 tokens kept — the finding-out is not";
    } });
    steps.push({ at: 9400, run: function () {
      title.textContent = "Add retry to the payments client";
      foot.textContent = "";
      show(recall);
    } });
    ACT3.forEach(function (_, i) {
      steps.push({ at: 10300 + i * 520, run: function () { show(a3[i]); } });
    });
    steps.push({ at: 12600, run: function () { foot.textContent = "right on the first attempt"; } });
    steps.push({ at: 16200, run: reset });

    var tl = Timeline(steps, 16600);

    if (REDUCED) {
      show(recall); show(lesson); a3.forEach(show);
      title.textContent = "Add retry to the payments client";
      foot.textContent = "right on the first attempt";
      return;
    }

    var frame = 0;
    setInterval(function () { spin.textContent = SPIN[frame++ % SPIN.length]; }, 260);

    var replay = root.querySelector("[data-replay]");
    if (replay) replay.addEventListener("click", function () { reset(); tl.play(); });

    new IntersectionObserver(function (e) {
      e[0].isIntersecting ? tl.play() : tl.stop();
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
    var card = root.querySelector("[data-morph-card]");
    var text = root.querySelector("[data-morph-text]");
    var lv = root.querySelector("[data-morph-lv]");
    var tok = root.querySelector("[data-morph-tok]");
    var bar = root.querySelector("[data-morph-bar]");
    var drop = root.querySelector("[data-morph-drop]");
    var beat = root.querySelector("[data-morph-beat]");
    var dots = root.querySelectorAll("[data-morph-step]");
    var i = 0, raf = null;

    function count(from, to) {
      if (REDUCED) { tok.textContent = to; return; }
      var t0 = performance.now();
      cancelAnimationFrame(raf);
      (function tick(now) {
        var k = Math.min(1, (now - t0) / 520);
        tok.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
        if (k < 1) raf = requestAnimationFrame(tick);
      })(t0);
    }

    function go(n) {
      var prev = LEVELS[i];
      i = ((n % LEVELS.length) + LEVELS.length) % LEVELS.length;
      var s = LEVELS[i];

      lv.textContent = s.lv;
      bar.style.width = s.pct + "%";
      count(Number(tok.textContent) || prev.tok, s.tok);

      text.style.opacity = "0";
      setTimeout(function () {
        morph(card, function () { text.textContent = s.text; }, 480);
        text.style.opacity = "1";
      }, REDUCED ? 0 : 200);

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
