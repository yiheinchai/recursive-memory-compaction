/* Two animations that carry the argument, so the page does not have to.
 *
 * 1. A terminal that plays one session, distils it to a single line, and then
 *    starts a NEW session with that line already in context. Terminal chrome
 *    does the explaining for free: nobody needs to be told it is a CLI session.
 *
 * 2. One lesson card that rewrites itself shorter, twice.
 *
 * Both use fixed-height slots. A figure that grows while it plays would shove
 * the rest of the page down on every tick, which is worse than the motion is
 * good. The shrinking happens inside a reserved box.
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

  /* A cancellable list of timed steps that can be replayed from zero. */
  function Timeline(steps, loopAt) {
    var handles = [];
    return {
      stop: function () {
        handles.forEach(clearTimeout);
        handles = [];
      },
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
  // 1. The terminal                                                         //
  // ======================================================================= //

  var ACT1 = [
    ["you",   "❯", "the S3 retry keeps silently failing"],
    ["agent", "⏺", "Checking the response status code…"],
    ["tool",  "└", "Bash  curl -sI $BUCKET/objects/42"],
    ["agent", "⏺", "Status is 200. Adding two more attempts…"],
    ["agent", "⏺", "Still failing. Adding logging…"],
    ["tool",  "└", "Edit  src/storage/s3.ts"],
    ["you",   "❯", "S3 returns 200 with the error in the body"],
    ["agent", "⏺", "Parsing the body instead. Fixed."]
  ];

  var ACT3 = [
    ["you",   "❯", "add retry to the payments client"],
    ["agent", "⏺", "Idempotency key set; parsing the response body."],
    ["tool",  "└", "Edit  src/payments/client.ts"],
    ["agent", "⏺", "Done — first try."]
  ];

  var LESSON = "Retry idempotent calls; parse bodies, not status codes.";

  function line(kind, mark, text) {
    var el = h("div", "ln " + kind);
    el.appendChild(h("span", "who", mark));
    el.appendChild(h("span", "txt", text));
    return el;
  }

  function terminal(root) {
    var bar = root.querySelector("[data-term-act]");
    var body = root.querySelector("[data-term-body]");
    var foot = root.querySelector("[data-term-foot]");

    var a1 = ACT1.map(function (l) { return line(l[0], l[1], l[2]); });
    var ctx = h("div", "ln rmc");
    ctx.appendChild(h("span", "who", "⋯"));
    ctx.appendChild(h("span", "txt", "RMC · 1 lesson recalled · 146 tok"));
    var lesson = line("lesson", "", LESSON);
    var a3 = ACT3.map(function (l) { return line(l[0], l[1], l[2]); });

    a1.concat([ctx, lesson], a3).forEach(function (n) { body.appendChild(n); });

    function show(n) { n.classList.remove("out"); n.classList.add("on"); }
    function hide(n) {
      if (!n.classList.contains("on")) return;
      n.classList.add("out");
      setTimeout(function () { n.classList.remove("on", "out"); }, 320);
    }
    function reset() {
      a1.concat([ctx, lesson], a3).forEach(function (n) {
        n.classList.remove("on", "out");
      });
      bar.textContent = "SESSION · MONDAY";
      foot.textContent = "";
    }

    var steps = [];
    ACT1.forEach(function (_, i) {
      steps.push({ at: 500 + i * 400, run: function () { show(a1[i]); } });
    });
    steps.push({ at: 3900, run: function () { foot.textContent = "4,200 tokens to get there"; } });
    steps.push({ at: 4900, run: function () { bar.textContent = "RMC · REFLECTING…"; } });
    steps.push({ at: 5700, run: function () {
      a1.forEach(hide);
      setTimeout(function () { show(lesson); }, 260);
    } });
    steps.push({ at: 6300, run: function () {
      bar.textContent = "LESSON · KEPT";
      foot.textContent = "146 tokens — the finding-out is not kept";
    } });
    steps.push({ at: 8600, run: function () {
      bar.textContent = "SESSION · THURSDAY";
      foot.textContent = "";
      show(ctx);
    } });
    ACT3.forEach(function (_, i) {
      steps.push({ at: 9400 + i * 430, run: function () { show(a3[i]); } });
    });
    steps.push({ at: 11400, run: function () { foot.textContent = "right on the first attempt"; } });
    steps.push({ at: 14600, run: reset });

    var tl = Timeline(steps, 15000);
    if (REDUCED) { show(ctx); show(lesson); a3.forEach(show); return; }

    var replay = root.querySelector("[data-replay]");
    if (replay) replay.addEventListener("click", function () { reset(); tl.play(); });

    new IntersectionObserver(function (e) {
      e[0].isIntersecting ? tl.play() : tl.stop();
    }, { threshold: 0.3 }).observe(root);
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
            "backoff at 100 ms, 400 ms and 1.6 s."
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
    var i = 0, tweenRaf = null;

    function count(from, to) {
      if (REDUCED) { tok.textContent = to; return; }
      var t0 = performance.now();
      cancelAnimationFrame(tweenRaf);
      (function tick(now) {
        var k = Math.min(1, (now - t0) / 520);
        tok.textContent = Math.round(from + (to - from) * (1 - Math.pow(1 - k, 3)));
        if (k < 1) tweenRaf = requestAnimationFrame(tick);
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
