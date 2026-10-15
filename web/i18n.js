/** Shared ZH/EN UI strings for demo pages. Long narrative stays in the report. */
(function (global) {
  const KEY = "blade_lang";
  const dict = {
    zh: {
      "nav.review": "评审 15 对",
      "nav.match": "锈签名 2017→2018",
      "nav.bg": "背景站位 2018→2017",
      "nav.stage": "阶段调试",
      "nav.stageShort": "阶段 1",
      "nav.matchShort": "锈签名",
      "nav.bgShort": "背景（可选）",
      "nav.reviewHere": "评审（本页）",
      "lang.toggle": "EN",
      "lang.title": "Switch to English",

      "index.title": "blade-revisit · 分阶段调试",
      "index.h1": "blade-revisit · 分阶段调试",
      "index.pick": "选图（阶段 1 单张）",
      "index.year": "年份",
      "index.file": "文件",
      "index.loadingNear": "近景筛选加载中…",
      "index.usableOnly": "排除边不可读（0687 类）",
      "index.run": "跑 1.0 + 1.1 + 1.2",
      "index.litBanner": "跑完后按 <strong>sunny / overcast</strong> 显隐视图按钮",
      "index.litOc": "工况 <strong>overcast</strong> · 显示阴天剥天 / 1.3 闭合 · 尖端已藏",
      "index.litSu": "工况 <strong>sunny</strong> · 显示晴天水禁 / tip · 1.3 已藏",
      "index.searchBox": "跨年检索（锈签名主线）",
      "index.searchHint":
        "主线：<strong>阴天 2017 → 晴天 2018</strong>（四点锈 crop）。请用 <a href=\"/match\" style=\"color:var(--accent)\">/match</a>；下方旧 ORB 仅留档。",
      "index.legacySummary": "展开（旧 ORB 2018→2017 · 非主线）",
      "index.refLimit": "2017 usable 池上限（0=全池）",
      "index.topK": "Top-K",
      "index.legacySearch": "旧路径 2018→2017",
      "index.bootHint":
        "当前主线是 <strong>1.0 光照</strong>。金样：2018 DJI_0190 ↔ 2017 DJI_0752。",
      "index.bootMeta": "选图后点「跑 1.0 + 1.1 + 1.2」。",
      "index.branchOc": "阴天线：1.0 → 1.1 内填 → 1.3 → mask → 1.2",
      "index.branchSu": "晴天线：1.0 → 1.1 → tip → mask → 1.2",
      "index.branchIdle": "先看「1.0 光照」；列表默认 usable_only。",
      "index.usableReject": "可用门控：REJECT",
      "index.usableKeep": "可用门控：KEEP",
      "index.view": "视图",
      "index.download": "下载当前视图",
      "index.fail": "失败：",
      "index.searchNeed2018": "检索请选 2018 作为 query",
      "index.searching": "检索中…",
      "index.searchDone": "对照完成",
      "index.nearHint": "近景",
      "index.usableHint": "可用",
      "index.total": "全库",
      "index.rejected": "拒识",
      "index.far": "远景",
      "index.cached": "缓存",
      "index.shots": " 张",

      "stage.1.0.title": "光照工况",
      "stage.1.0.blurb": "sunny/overcast · 先调试图不改 mask",
      "stage.1.1.title": "整机 mask",
      "stage.1.1.blurb": "边缘填充 + 轮毂连通",
      "stage.1.2.title": "机上特征点",
      "stage.1.2.blurb": "只在 mask 内提点",

      "match.title": "blade-revisit · 锈签名检索",
      "match.h1": "锈签名检索 · 2017 → 2018",
      "match.pick": "选 2017 查询图（阴天锈更显）",
      "match.file": "文件",
      "match.loading": "加载中…",
      "match.usableOnly": "query 列表只用 usable",
      "match.refLimit": "2018 usable 池上限（0=全池）",
      "match.topK": "Top-K",
      "match.search": "锈签名检索",
      "match.hint":
        "主线：阴天 <strong>2017 query</strong> 提四点锈 → 搜晴天 <strong>2018 usable</strong>。金样期望 <strong>0752 → 0190 Top-1</strong>。看 rust crop，不是全图 ORB。",
      "match.boot": "选 2017（默认 DJI_0752）→ 检索 → 点 Top-k 看 rust↔rust。",
      "match.idle": "尚未检索。",
      "match.gold1": "金样",
      "match.goldMiss": "金样未进榜",
      "match.clickCrop": "点一条看 rust crop。",
      "match.searching":
        "检索中… 首次建 2018 签名会慢；有缓存后很快。",
      "match.fail": "失败：",
      "match.locusQ": "2017 锈在原图哪里",
      "match.locusR": "2018 锈在原图哪里",
      "match.cropCap": "rust↔rust：上定位 · 下 crop（主证据）",
      "match.noPreview": "无 rust preview",
      "match.provenance":
        "青/绿圈=锈四点 · 橙框≈crop 来源（可分析性优先）",
      "match.galleryLine": " 张 · gallery 固定 2018 usable",

      "view.original": "原图",
      "view.lighting": "1.0 光照",
      "view.sky_water": "1.0b 天/水抠除",
      "view.sunny_water": "1.0b 晴天水禁",
      "view.edge_seed": "1.0c 边障+种子",
      "view.seed": "1.1a 几何种子",
      "view.seed_aux": "1.1a 辅助线",
      "view.refine_delta": "1.1b 内填/白扩",
      "view.geo2": "1.1c 二步几何",
      "view.tip_need": "1.1d 尖端判断",
      "view.tip_edge": "1.1e 叶尖边缘",
      "view.expand_lines": "1.3a 扩线(3a-3d)",
      "view.expand_trap": "1.3c 梯形探边",
      "view.expand_delta": "1.3b 闭合填",
      "view.overlay": "1.1 最终 mask",
      "view.blade": "1.1 只风机",
      "view.kps_masked": "1.2 mask 内点",
      "view.kps_full": "对照·全图点",
      "view.seed_aux_title": "已证伪，默认隐藏",

      "bg.title": "blade-revisit · 背景站位",
      "bg.h1": "背景站位 · 2018 → 2017",
      "bg.pick": "选 2018 查询图（晴天背景更清楚）",
      "bg.file": "文件",
      "bg.loading": "加载中…",
      "bg.method": "方法",
      "bg.methodBlocks": "blocks · 色块多锚投票（v2）",
      "bg.methodHsv": "hsv · 网格直方图（v1）",
      "bg.refLimit": "2017 usable 上限（0=全池）",
      "bg.topK": "Top-K",
      "bg.search": "背景站位检索",
      "bg.hint":
        "默认 <strong>色块投票</strong>：类+方位+边缘档。签名缓存落盘；只有 miss&gt;0 才重跑 mask。",
      "bg.boot": "选 2018（默认 DJI_0190）→ 检索 → 看 2017 Top-k 是否像附近站位。",
      "bg.idle": "尚未检索。",
      "bg.gold": "金样",
      "bg.goldMiss": "金样未进榜",
      "bg.fail": "失败：",
      "bg.searching": "检索中…",
      "bg.galleryLine": " 张 · gallery=2017 usable",
      "bg.cacheNote": "（miss=0 表示 gallery 已落盘）",

      "review.title": "blade-revisit · 隔年锈迹评审",
      "review.h1": "隔年锈迹评审 · 2017 → 2018",
      "review.sub":
        "主打叶片锈纹理对齐；系统只给下一步建议（重拍 / 拒绝 / 再看 / 进人），<strong style=\"color:var(--warn)\">不鉴定是否生锈、不给维修结论</strong>。",
      "review.banner":
        "<strong>15 对</strong>可在左侧改期望档；锈主线 = 2017→2018。背景为可选旁证，不改档。叙述与论证见报告，不在本页展开。",
      "review.listTitle": "评测 15 对",
      "review.loading": "加载中…",
      "review.boot": "点左侧一对开始。默认看锈证据；可点步骤回放。",
      "review.provisional": "临时表",
      "review.listMeta": " 对 · 可改档后保存",
      "review.humanNote": "人注：",
      "review.runRust": "跑锈签名（这对）",
      "review.runBg": "可选：背景站位参考（2018→2017）",
      "review.expect": "期望下一步",
      "review.verdictHelp":
        "叶片纹理对齐后比较相对变化；系统不输出「已确认生锈 / 要不要修」。背景分仅供参考。",
      "review.editTier": "改期望档",
      "review.note": "备注",
      "review.save": "保存本对期望",
      "review.bgLoaded": "背景参考（可选）",
      "review.bgNoTier": "不改档",
      "review.bgIdle": "背景站位参考未加载。点上方可选按钮；仅旁证。",
      "review.bgCap": "2018→2017 背景块对照",
      "review.step.side": "① 原图+锈位",
      "review.step.lighting": "② 1.0 光照",
      "review.step.overlay": "③ 1.1 mask",
      "review.step.kps": "④ 1.2 提点",
      "review.step.rust": "⑤ 锈 crop",
      "review.tier.retake": "重拍",
      "review.tier.reject": "拒绝配成同一处",
      "review.tier.watch": "明年再看",
      "review.tier.escalate": "进人复核",
      "review.tier.skip": "跳过检索",
      "review.sideHint":
        "圈点=锈四点 · 橙框≈crop 从原图裁的位置（分析用；多数对期望拒绝）",
      "review.src2017": "2017 来源",
      "review.src2018": "2018 来源",
      "review.noCachePreview": "无缓存预览，点「跑锈签名」生成",
      "review.needRustRun": "尚无 rust preview，点「跑锈签名（这对）」。",
      "review.rustFail": "锈检索失败：",
      "review.bgFail": "背景参考失败：",
    },
    en: {
      "nav.review": "Review 15 pairs",
      "nav.match": "Rust search 2017→2018",
      "nav.bg": "Background prior 2018→2017",
      "nav.stage": "Stage debug",
      "nav.stageShort": "Stage 1",
      "nav.matchShort": "Rust search",
      "nav.bgShort": "Background (optional)",
      "nav.reviewHere": "Review (this page)",
      "lang.toggle": "中文",
      "lang.title": "切换到中文",

      "index.title": "blade-revisit · stage debug",
      "index.h1": "blade-revisit · stage debug",
      "index.pick": "Pick a frame (stage 1)",
      "index.year": "Year",
      "index.file": "File",
      "index.loadingNear": "Loading near-shot list…",
      "index.usableOnly": "Drop edge-unreadable (e.g. 0687)",
      "index.run": "Run 1.0 + 1.1 + 1.2",
      "index.litBanner":
        "After run, views follow <strong>sunny / overcast</strong>",
      "index.litOc":
        "Lighting <strong>overcast</strong> · sky peel / 1.3 close shown · tip hidden",
      "index.litSu":
        "Lighting <strong>sunny</strong> · water ban / tip shown · 1.3 hidden",
      "index.searchBox": "Cross-year search (rust main line)",
      "index.searchHint":
        "Main line: <strong>overcast 2017 → sunny 2018</strong> (4-point rust crop). Use <a href=\"/match\" style=\"color:var(--accent)\">/match</a>; legacy ORB below is archive-only.",
      "index.legacySummary": "Expand (legacy ORB 2018→2017 · not main line)",
      "index.refLimit": "2017 usable pool cap (0=all)",
      "index.topK": "Top-K",
      "index.legacySearch": "Legacy path 2018→2017",
      "index.bootHint":
        "Main line starts at <strong>1.0 lighting</strong>. Gold: 2018 DJI_0190 ↔ 2017 DJI_0752.",
      "index.bootMeta": "Pick a frame, then run 1.0 + 1.1 + 1.2.",
      "index.branchOc": "Overcast: 1.0 → fill → 1.3 → mask → 1.2",
      "index.branchSu": "Sunny: 1.0 → 1.1 → tip → mask → 1.2",
      "index.branchIdle": "Start at 1.0 lighting; list defaults to usable_only.",
      "index.usableReject": "Usable gate: REJECT",
      "index.usableKeep": "Usable gate: KEEP",
      "index.view": "View",
      "index.download": "Download current view",
      "index.fail": "Failed: ",
      "index.searchNeed2018": "Pick 2018 as query for legacy search",
      "index.searching": "Searching…",
      "index.searchDone": "Done",
      "index.nearHint": "Near",
      "index.usableHint": "Usable",
      "index.total": "all",
      "index.rejected": "rejected",
      "index.far": "far",
      "index.cached": "cached",
      "index.shots": " shots",

      "stage.1.0.title": "Lighting",
      "stage.1.0.blurb": "sunny/overcast · debug views only",
      "stage.1.1.title": "Turbine mask",
      "stage.1.1.blurb": "Edge fill + hub connectivity",
      "stage.1.2.title": "On-mask keypoints",
      "stage.1.2.blurb": "Detect only inside the mask",

      "match.title": "blade-revisit · rust signature search",
      "match.h1": "Rust signature · 2017 → 2018",
      "match.pick": "Pick 2017 query (rust clearer under overcast)",
      "match.file": "File",
      "match.loading": "Loading…",
      "match.usableOnly": "Usable-only query list",
      "match.refLimit": "2018 usable pool cap (0=all)",
      "match.topK": "Top-K",
      "match.search": "Run rust search",
      "match.hint":
        "Main line: overcast <strong>2017 query</strong> 4 rust loci → sunny <strong>2018 usable</strong>. Gold: <strong>0752 → 0190 Top-1</strong>. Compare rust crops, not full-frame ORB.",
      "match.boot": "Pick 2017 (default DJI_0752) → search → click Top-k for rust↔rust.",
      "match.idle": "No search yet.",
      "match.gold1": "Gold",
      "match.goldMiss": "Gold not in Top-k",
      "match.clickCrop": "Click a row for the rust crop.",
      "match.searching":
        "Searching… first-time 2018 signatures are slow; then cached.",
      "match.fail": "Failed: ",
      "match.locusQ": "2017 rust loci on original",
      "match.locusR": "2018 rust loci on original",
      "match.cropCap": "rust↔rust: loci above · crop below (main evidence)",
      "match.noPreview": "No rust preview",
      "match.provenance":
        "Cyan/green = 4 rust loci · orange ≈ crop window (analyzability first)",
      "match.galleryLine": " · gallery fixed to 2018 usable",

      "view.original": "Original",
      "view.lighting": "1.0 lighting",
      "view.sky_water": "1.0b sky/water peel",
      "view.sunny_water": "1.0b sunny water ban",
      "view.edge_seed": "1.0c edge+seed",
      "view.seed": "1.1a geometry seed",
      "view.seed_aux": "1.1a aux lines",
      "view.refine_delta": "1.1b fill/grow",
      "view.geo2": "1.1c geo step-2",
      "view.tip_need": "1.1d tip need",
      "view.tip_edge": "1.1e tip edges",
      "view.expand_lines": "1.3a expand lines",
      "view.expand_trap": "1.3c trap edges",
      "view.expand_delta": "1.3b close fill",
      "view.overlay": "1.1 final mask",
      "view.blade": "1.1 turbine only",
      "view.kps_masked": "1.2 masked kps",
      "view.kps_full": "ctrl · full-frame kps",
      "view.seed_aux_title": "Falsified; hidden by default",

      "bg.title": "blade-revisit · background prior",
      "bg.h1": "Background prior · 2018 → 2017",
      "bg.pick": "Pick 2018 query (background clearer in sun)",
      "bg.file": "File",
      "bg.loading": "Loading…",
      "bg.method": "Method",
      "bg.methodBlocks": "blocks · multi-anchor vote (v2)",
      "bg.methodHsv": "hsv · grid histogram (v1)",
      "bg.refLimit": "2017 usable cap (0=all)",
      "bg.topK": "Top-K",
      "bg.search": "Run background search",
      "bg.hint":
        "Default <strong>block voting</strong>: class + bearing + edge bin. Signatures on disk; remask only when miss&gt;0.",
      "bg.boot": "Pick 2018 (default DJI_0190) → search → check if 2017 Top-k looks nearby.",
      "bg.idle": "No search yet.",
      "bg.gold": "Gold",
      "bg.goldMiss": "Gold not in Top-k",
      "bg.fail": "Failed: ",
      "bg.searching": "Searching…",
      "bg.galleryLine": " · gallery=2017 usable",
      "bg.cacheNote": " (miss=0 → gallery already on disk)",

      "review.title": "blade-revisit · revisit review",
      "review.h1": "Revisit review · 2017 → 2018",
      "review.sub":
        "Blade rust texture alignment; the system only suggests next step (retake / reject / watch / escalate). <strong style=\"color:var(--warn)\">No rust diagnosis, no repair decision.</strong>",
      "review.banner":
        "<strong>15 pairs</strong>: edit expected tier on the left. Rust main line = 2017→2018. Background is optional side evidence. Full narrative lives in the report, not this UI.",
      "review.listTitle": "Eval 15 pairs",
      "review.loading": "Loading…",
      "review.boot": "Pick a pair. Default shows rust evidence; steps replay mask/keypoints.",
      "review.provisional": "provisional",
      "review.listMeta": " pairs · edit tier then save",
      "review.humanNote": "human note: ",
      "review.runRust": "Run rust signature (this pair)",
      "review.runBg": "Optional: background prior (2018→2017)",
      "review.expect": "Expected next step",
      "review.verdictHelp":
        "After texture alignment, compare relative change. No “confirmed rust / repair” output. Background score is advisory only.",
      "review.editTier": "Edit expected tier",
      "review.note": "Note",
      "review.save": "Save expected tier",
      "review.bgLoaded": "Background (optional)",
      "review.bgNoTier": "does not set tier",
      "review.bgIdle": "Background prior not loaded. Optional button above; advisory only.",
      "review.bgCap": "2018→2017 background blocks",
      "review.step.side": "① Original + rust loci",
      "review.step.lighting": "② 1.0 lighting",
      "review.step.overlay": "③ 1.1 mask",
      "review.step.kps": "④ 1.2 keypoints",
      "review.step.rust": "⑤ Rust crop",
      "review.tier.retake": "Retake",
      "review.tier.reject": "Reject same-site match",
      "review.tier.watch": "Watch next year",
      "review.tier.escalate": "Escalate to human",
      "review.tier.skip": "Skip retrieval",
      "review.sideHint":
        "Circles = 4 rust loci · orange ≈ crop cut from original (for analysis; most pairs expect reject)",
      "review.src2017": "2017 source",
      "review.src2018": "2018 source",
      "review.noCachePreview": "No cached preview — run rust signature",
      "review.needRustRun": "No rust preview yet — run rust signature for this pair.",
      "review.rustFail": "Rust search failed: ",
      "review.bgFail": "Background prior failed: ",
    },
  };

  // Default English for public review; localStorage wins after first toggle.
  let lang = "en";
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "en" || saved === "zh") lang = saved;
  } catch (_) {}

  const listeners = [];

  function t(key) {
    const pack = dict[lang] || dict.zh;
    return pack[key] != null ? pack[key] : (dict.zh[key] != null ? dict.zh[key] : key);
  }

  function getLang() {
    return lang;
  }

  function setLang(next) {
    lang = next === "en" ? "en" : "zh";
    try {
      localStorage.setItem(KEY, lang);
    } catch (_) {}
    document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
    applyDom();
    listeners.forEach((fn) => {
      try {
        fn(lang);
      } catch (_) {}
    });
  }

  function onChange(fn) {
    listeners.push(fn);
  }

  function applyDom() {
    document.querySelectorAll("[data-i18n]").forEach((el) => {
      const key = el.getAttribute("data-i18n");
      if (!key) return;
      const mode = el.getAttribute("data-i18n-mode") || "text";
      const val = t(key);
      if (mode === "html") el.innerHTML = val;
      else if (mode === "title") el.setAttribute("title", val);
      else el.textContent = val;
    });
    document.querySelectorAll("[data-i18n-title]").forEach((el) => {
      el.setAttribute("title", t(el.getAttribute("data-i18n-title")));
    });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
      el.setAttribute("placeholder", t(el.getAttribute("data-i18n-placeholder")));
    });
    const titleEl = document.querySelector("title[data-i18n]");
    if (titleEl) document.title = t(titleEl.getAttribute("data-i18n"));
    const btn = document.getElementById("langToggle");
    if (btn) {
      btn.textContent = t("lang.toggle");
      btn.setAttribute("title", t("lang.title"));
    }
  }

  function mountToggle(parent) {
    if (!parent || document.getElementById("langToggle")) return;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.id = "langToggle";
    btn.textContent = t("lang.toggle");
    btn.title = t("lang.title");
    btn.style.cssText =
      "width:auto;padding:0.3rem 0.55rem;font-size:0.75rem;cursor:pointer;" +
      "background:#2a2f38;border:1px solid #3c424d;color:var(--text);border-radius:4px;";
    btn.addEventListener("click", () => setLang(lang === "zh" ? "en" : "zh"));
    parent.appendChild(btn);
  }

  global.BladeI18n = { t, getLang, setLang, onChange, applyDom, mountToggle };
  document.documentElement.lang = lang === "en" ? "en" : "zh-CN";
})(window);
