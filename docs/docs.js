/* Documentation behaviour: search, copy buttons, theme, scrollspy, mobile nav.
 *
 * No framework and no build step. The whole file is smaller than one npm
 * dependency's licence header, and it has to keep working for the reader whose
 * tooling is already broken — which is often why they are here. */

(function () {
  'use strict';

  var $ = function (sel, root) { return (root || document).querySelector(sel); };
  var $$ = function (sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  };

  /* ------------------------------------------------------------ theme */
  var root = document.documentElement;
  var themeBtn = $('.theme');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = root.dataset.theme === 'dark' ? 'light' : 'dark';
      root.dataset.theme = next;
      try { localStorage.setItem('rmc-theme', next); } catch (e) {}
    });
  }

  /* ------------------------------------------------------- mobile nav */
  var burger = $('.burger');
  var side = $('#side');
  if (burger && side) {
    burger.addEventListener('click', function () {
      var open = side.classList.toggle('open');
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
    // Tapping a link should navigate, not leave the drawer covering the page
    // you just navigated to.
    side.addEventListener('click', function (e) {
      if (e.target.tagName === 'A') {
        side.classList.remove('open');
        burger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  /* ----------------------------------------------------- copy buttons */
  // Every code block in these docs is something the reader is meant to run.
  // Selecting a multi-line block by hand is a small tax paid on every visit.
  $$('.prose pre').forEach(function (pre) {
    var btn = document.createElement('button');
    btn.className = 'copy';
    btn.type = 'button';
    btn.textContent = 'Copy';
    btn.addEventListener('click', function () {
      // The comment spans are annotation, not part of the command — copying
      // them gives you something that does not run.
      var clone = pre.cloneNode(true);
      $$('.c', clone).forEach(function (c) { c.remove(); });
      var text = clone.textContent.replace(/[ \t]+$/gm, '').trim();
      navigator.clipboard.writeText(text).then(function () {
        btn.textContent = 'Copied';
        btn.classList.add('done');
        setTimeout(function () {
          btn.textContent = 'Copy';
          btn.classList.remove('done');
        }, 1400);
      }).catch(function () { btn.textContent = 'Press ⌘C'; });
    });
    pre.appendChild(btn);
  });

  /* --------------------------------------------------------- scrollspy */
  var marks = $$('.onpage a');
  if (marks.length) {
    var targets = marks.map(function (a) {
      return document.getElementById(a.getAttribute('href').slice(1));
    });
    var spy = function () {
      // The heading whose top has most recently passed under the sticky bar.
      // Comparing against the viewport centre instead makes short final
      // sections unreachable — they never occupy it.
      var best = 0;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i] && targets[i].getBoundingClientRect().top <= 96) best = i;
      }
      marks.forEach(function (a, i) { a.classList.toggle('here', i === best); });
    };
    addEventListener('scroll', spy, { passive: true });
    spy();
  }

  /* ------------------------------------------------------------ search */
  var data = [];
  try { data = JSON.parse($('#rmc-search').textContent); } catch (e) {}

  var box = $('#searchbox');
  var input = $('#q');
  var results = $('#results');
  var chosen = 0;

  function open() {
    if (!box) return;
    box.hidden = false;
    input.value = '';
    render([]);
    input.focus();
  }
  function close() { if (box) box.hidden = true; }

  function score(rec, terms) {
    // Deliberately crude: substring hits, weighted by where they land. A real
    // ranker would be better and would also be a dependency; for a site of
    // eighteen pages the heading match is almost always the right answer.
    var head = (rec.h || '').toLowerCase();
    var page = (rec.p || '').toLowerCase();
    var body = (rec.t || '').toLowerCase();
    var total = 0;
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i];
      if (!t) continue;
      var hit = 0;
      if (head.indexOf(t) === 0) hit += 60;
      else if (head.indexOf(t) > -1) hit += 40;
      if (page.indexOf(t) > -1) hit += 8;
      if (body.indexOf(t) > -1) hit += 6;
      if (!hit) return 0;          // every term must appear somewhere
      total += hit;
    }
    return total;
  }

  function render(list) {
    if (!results) return;
    if (!list.length) {
      results.innerHTML = input && input.value.trim()
        ? '<div class="empty">No matches.</div>'
        : '<div class="empty">Search commands, settings and concepts.</div>';
      return;
    }
    results.innerHTML = list.map(function (r, i) {
      return '<a class="' + (i === chosen ? 'sel' : '') + '" href="' + r.u + '">' +
        '<div class="h">' + esc(r.h) + '</div>' +
        '<div class="p">' + esc(r.p) + '</div></a>';
    }).join('');
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  var current = [];
  function run() {
    var terms = input.value.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { current = []; chosen = 0; render([]); return; }
    current = data
      .map(function (r) { return { r: r, s: score(r, terms) }; })
      .filter(function (x) { return x.s > 0; })
      .sort(function (a, b) { return b.s - a.s; })
      .slice(0, 12)
      .map(function (x) { return x.r; });
    chosen = 0;
    render(current);
  }

  if (input) {
    input.addEventListener('input', run);
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        if (!current.length) return;
        chosen = (chosen + (e.key === 'ArrowDown' ? 1 : -1) + current.length) % current.length;
        render(current);
        var sel = $('#results a.sel');
        if (sel) sel.scrollIntoView({ block: 'nearest' });
      } else if (e.key === 'Enter' && current[chosen]) {
        location.href = current[chosen].u;
      } else if (e.key === 'Escape') {
        close();
      }
    });
  }

  var opener = $('.search-open');
  if (opener) opener.addEventListener('click', open);
  if (box) box.addEventListener('click', function (e) { if (e.target === box) close(); });

  addEventListener('keydown', function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (!typing && (e.key === '/' || ((e.metaKey || e.ctrlKey) && e.key === 'k'))) {
      e.preventDefault();
      open();
    } else if (e.key === 'Escape') {
      close();
    }
  });
})();
