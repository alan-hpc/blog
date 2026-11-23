(function () {
  function installMathStyle() {
    if (document.getElementById("readable-math-style")) return;
    const style = document.createElement("style");
    style.id = "readable-math-style";
    style.textContent = [
      ".zh .katex,.toc-title .katex,.section-h .katex{font-size:1em}",
      ".zh .katex{margin:0 .03em}",
      ".zh .katex-display{margin:.45em 0 .55em}",
    ].join("\n");
    document.head.appendChild(style);
  }

  function idToLatex(id) {
    const s = String(id).replace(/\s+/g, "");
    const map = {
      Lrow: "L_{\\mathrm{row}}",
      Lcol: "L_{\\mathrm{col}}",
      Ltiled: "L_{\\mathrm{tiled}}",
      Atiled: "A_{\\mathrm{tiled}}",
      Lf: "L_f",
      Lg: "L_g",
      fL: "f_L",
      gL: "g_L",
      Ac: "A^c",
      Bc: "B^c",
      idX: "\\mathrm{id}_X",
      idY: "\\mathrm{id}_Y",
      idS: "\\mathrm{id}_S",
      idT: "\\mathrm{id}_T",
    };
    if (map[s]) return map[s];
    if (/^[A-Za-z][0-9]+$/.test(s)) return s.replace(/^([A-Za-z])([0-9]+)$/, "$1_$2");
    return s.replace(/♭/g, "^{\\flat}").replace(/′/g, "'");
  }

  function toLatex(expr) {
    let out = String(expr).trim();
    out = out
      .replace(/\s*:\s*/g, ":")
      .replace(/\s*,\s*/g, ",\\,")
      .replace(/→/g, "\\to ")
      .replace(/↠/g, "\\twoheadrightarrow ")
      .replace(/◦/g, "\\circ ")
      .replace(/⊗/g, "\\otimes ")
      .replace(/⊘/g, "\\oslash ")
      .replace(/⊕/g, "\\oplus ")
      .replace(/⊥/g, "\\perp ")
      .replace(/⋆/g, "\\star ")
      .replace(/·/g, "\\cdot ")
      .replace(/≥/g, "\\ge ")
      .replace(/≤/g, "\\le ")
      .replace(/≠/g, "\\ne ")
      .replace(/∈/g, "\\in ")
      .replace(/⊂/g, "\\subset ")
      .replace(/↦/g, "\\mapsto ")
      .replace(/×/g, "\\times ");

    out = out
      .replace(/\bsize\(/g, "\\mathrm{size}(")
      .replace(/\bcosize\(/g, "\\mathrm{cosize}(")
      .replace(/\brank\(/g, "\\mathrm{rank}(")
      .replace(/\blen\(/g, "\\mathrm{len}(")
      .replace(/\bmode(\d*)\(/g, function (_, n) {
        return "\\mathrm{mode}" + (n ? "_" + n : "") + "(";
      })
      .replace(/\bcoal♭\(/g, "\\mathrm{coal}^{\\flat}(")
      .replace(/\bcomp♭\(/g, "\\mathrm{comp}^{\\flat}(")
      .replace(/\bcomp\(/g, "\\mathrm{comp}(")
      .replace(/\bsort\(/g, "\\mathrm{sort}(")
      .replace(/\bsqueeze\(/g, "\\mathrm{squeeze}(");

    out = out
      .replace(/\bLrow\b/g, "L_{\\mathrm{row}}")
      .replace(/\bLcol\b/g, "L_{\\mathrm{col}}")
      .replace(/\bLtiled\b/g, "L_{\\mathrm{tiled}}")
      .replace(/\bAtiled\b/g, "A_{\\mathrm{tiled}}")
      .replace(/\bLf\b/g, "L_f")
      .replace(/\bLg\b/g, "L_g")
      .replace(/\bfL\b/g, "f_L")
      .replace(/\bgL\b/g, "g_L")
      .replace(/\bAc\b/g, "A^c")
      .replace(/\bBc\b/g, "B^c")
      .replace(/\b([A-Z])([0-9]+)\b/g, "$1_$2")
      .replace(/\bΦ([A-Za-z][A-Za-z0-9♭]*)/g, function (_, x) {
        return "\\Phi_{" + idToLatex(x) + "}";
      })
      .replace(/\bφ([A-Za-z][A-Za-z0-9_♭]*)/g, function (_, x) {
        return "\\varphi_{" + idToLatex(x) + "}";
      })
      .replace(/Φ/g, "\\Phi")
      .replace(/φ/g, "\\varphi")
      .replace(/♭/g, "^{\\flat}");

    return out.replace(/\s+/g, " ").trim();
  }

  function normalizeSegment(segment) {
    const stash = [];
    const hold = function (expr) {
      const i = stash.push(toLatex(expr)) - 1;
      return "\uE000" + i + "\uE001";
    };

    let s = segment;

    s = s.replace(/\bLf\s*⊗\s*g\b/g, hold("L_{f⊗g}"));
    s = s.replace(/\bLf\s*⊘\s*g\b/g, hold("L_{f⊘g}"));

    s = s.replace(/\b(Lrow|Lcol|Ltiled|L|A|B[0-9]?)\s*=\s*\(([^()，。；]{1,50})\)\s*:\s*\(([^()，。；]{1,50})\)/g, function (_, lhs, a, b) {
      return hold(idToLatex(lhs) + "=(" + a + "):(" + b + ")");
    });

    s = s.replace(/\b([fgheFαβσι])\s*:\s*([A-Za-z0-9_\s,()⟨⟩∗*′'\\]+?)\s*(→|↠)\s*([A-Za-z0-9_\s,()⟨⟩∗*′'\\]+)/g, function (_, lhs, dom, arrow, cod) {
      return hold(lhs + ":" + dom + arrow + cod);
    });

    s = s.replace(/\b(?:rank|size|cosize|len|mode\d*)\([^)]+\)\s*(?:=|≠|≥|≤)\s*[^，。；。<\n]+/g, hold);
    s = s.replace(/[Φφ][A-Za-z][A-Za-z0-9_♭⋆]*(?:\([^)]*\))?\s*(?:=|≠|≥|≤|⊂|∈)\s*[^，。；。<\n]+/g, hold);

    s = s.replace(/([A-Za-zΑ-ωΦφ][A-Za-z0-9_♭′']*|\([A-Za-z0-9_,\s]+\))\s*[◦⊗⊘⊕⊥⋆]\s*([A-Za-zΑ-ωΦφ][A-Za-z0-9_♭′']*|\([A-Za-z0-9_,\s]+\))/g, hold);

    s = s.replace(/\b([0-9]+)\s*×\s*([0-9]+)\b/g, function (_, a, b) {
      return hold(a + "×" + b);
    });
    s = s.replace(/\(([ijkℓl0-9]+),\s*([ijkℓl0-9]+)\)/g, function (_, a, b) {
      return hold("(" + a.replace(/l/g, "\\ell") + "," + b.replace(/l/g, "\\ell") + ")");
    });
    s = s.replace(/\b([0-9]+)\s*i\s*\+\s*([0-9]+)\s*j\b/g, function (_, a, b) {
      return hold(a + "i+" + b + "j");
    });
    s = s.replace(/\bi\s*\+\s*([0-9]+)\s*j\b/g, function (_, b) {
      return hold("i+" + b + "j");
    });
    s = s.replace(/\b([0-9]+)\s*i\s*\+\s*j\b/g, function (_, a) {
      return hold(a + "i+j");
    });
    s = s.replace(/\ba,\s*b\b/g, hold("a,b"));

    s = s.replace(/\b(Lrow|Lcol|Ltiled|Atiled|Lf|Lg|fL|gL)\b/g, function (_, id) {
      return hold(idToLatex(id));
    });
    s = s.replace(/[Φφ][A-Za-z][A-Za-z0-9_♭⋆]*/g, hold);

    return s.replace(/\uE000(\d+)\uE001/g, function (_, i) {
      return "$" + stash[Number(i)] + "$";
    });
  }

  function normalizeText(text) {
    return String(text)
      .split(/(\$\$[\s\S]*?\$\$|\$[^$]*\$)/g)
      .map(function (part) {
        return part.startsWith("$") ? part : normalizeSegment(part);
      })
      .join("");
  }

  window.normalizeReadableMath = function (root) {
    root = root || document.body;
    installMathStyle();
    const targets = root.querySelectorAll("p.zh,.toc-title,.section-h,li,td,th");
    targets.forEach(function (el) {
      const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
        acceptNode: function (node) {
          const parent = node.parentElement;
          if (!parent || parent.closest("script,style,code,pre,.katex")) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        },
      });
      const nodes = [];
      let node;
      while ((node = walker.nextNode())) nodes.push(node);
      nodes.forEach(function (textNode) {
        const next = normalizeText(textNode.nodeValue);
        if (next !== textNode.nodeValue) textNode.nodeValue = next;
      });
    });
  };
})();
