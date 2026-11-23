/* ════════════════════════════════════════════════════════════════════════════
   reader.js  —  Yang Min 技术博客阅读器共享脚本
   所有 static/readers/*.html 共享此文件
   ════════════════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  /* ── Sidebar active link tracking ─────────────────────────────────────── */
  const reader = document.getElementById('reader');
  if (reader) {
    const sections = [...document.querySelectorAll(
      'section[id], h2[id], h3[id], h4[id], .section-h[id]'
    )];
    const links = [...document.querySelectorAll('#sb a[href^="#"]')];

    function updateActive() {
      const top = reader.scrollTop + 90;
      let cur = '';
      for (const s of sections) {
        if (s.offsetTop <= top) cur = '#' + s.id;
      }
      for (const a of links) {
        a.classList.toggle('active', a.getAttribute('href') === cur);
      }
    }
    reader.addEventListener('scroll', updateActive, { passive: true });
    updateActive();
  }

  /* ── Lightbox ──────────────────────────────────────────────────────────── */
  const lb = document.getElementById('lb');
  const lbImg = document.getElementById('lb-img');

  window.lbShow = function (src) {
    if (!lb || !lbImg) return;
    lbImg.src = src;
    lb.classList.add('open');
  };

  function lbHide() {
    if (!lb || !lbImg) return;
    lb.classList.remove('open');
    lbImg.src = '';
  }

  /* lbClose is used in onclick="lbClose(event)" on the lb overlay */
  window.lbClose = function (e) {
    if (!e || e.target === lb || (e.target && e.target.id === 'lb-close')) lbHide();
  };

  if (lb) {
    lb.addEventListener('click', function (e) {
      if (e.target === lb || e.target.id === 'lb-close') lbHide();
    });
  }
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') lbHide();
  });

  /* Wire up all figure images to the lightbox */
  document.querySelectorAll('figure img').forEach(function (img) {
    img.style.cursor = 'zoom-in';
    img.addEventListener('click', function () { lbShow(img.src); });
  });

  /* ── Highlight.js initialization ─────────────────────────────────────── */
  function initHljs() {
    if (typeof hljs === 'undefined') return;

    /* Highlight <pre><code> blocks */
    document.querySelectorAll('pre code').forEach(function (el) {
      /* Skip if already highlighted or contains Hugo chroma spans */
      if (el.classList.contains('hljs')) return;
      if (el.querySelector('span.cl, span.line')) {
        /* Strip Hugo chroma spans first */
        el.textContent = el.textContent;
      }
      hljs.highlightElement(el);
    });

    /* Handle bare <pre class="code-block"> without inner <code> */
    document.querySelectorAll('pre.code-block').forEach(function (pre) {
      if (pre.querySelector('code')) return;
      const c = document.createElement('code');
      c.textContent = pre.textContent;
      pre.textContent = '';
      pre.appendChild(c);
      hljs.highlightElement(c);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initHljs);
  } else {
    initHljs();
  }

  /* ── KaTeX auto-render ─────────────────────────────────────────────────── */
  function initKatex() {
    if (typeof renderMathInElement === 'undefined') return;
    const readerEl = document.getElementById('reader') || document.body;
    renderMathInElement(readerEl, {
      delimiters: [
        { left: '$$', right: '$$', display: true  },
        { left: '$',  right: '$',  display: false },
        { left: '\\[', right: '\\]', display: true  },
        { left: '\\(', right: '\\)', display: false }
      ],
      throwOnError: false,
      trust: false
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initKatex);
  } else {
    initKatex();
  }

})();

/* ════════════════════════════════════════════════════════════
   THEME — read pref-theme from localStorage (set by Hugo site
   theme-toggle) and apply to <html data-theme>.
   Runs as IIFE on script load (early enough to avoid FOUC).
   ════════════════════════════════════════════════════════════ */
(function () {
  try {
    var pref = localStorage.getItem('pref-theme');
    if (pref === 'dark') {
      document.documentElement.dataset.theme = 'dark';
    } else if (pref === 'light') {
      document.documentElement.dataset.theme = 'light';
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
      document.documentElement.dataset.theme = 'dark';
    }
  } catch (e) {}
})();
