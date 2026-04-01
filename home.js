let deferredPrompt = null;
const $ = (id) => document.getElementById(id);
const safeText = (id, txt) => { const el = $(id); if (el) el.textContent = txt; };
const strings = {
  en: {
    h1: "Habitat × Ops → Catch Probability",
    p1: "Scientific habitat/ops preview with explainable top hotspots.",
    launch: "Launch App",
    install: "Install PWA",
    prevTitle: "Latest preview"
  },
  fa: {
    h1: "زیستگاه × عملیات → احتمال صید",
    p1: "پیش‌نمایش علمی زیستگاه/عملیات با نقاط برتر توضیح‌پذیر.",
    launch: "ورود به اپ",
    install: "نصب اپ",
    prevTitle: "آخرین پیش‌نمایش"
  }
};
let lang = localStorage.getItem("lang") || "en";
function applyLang(){
  const t = strings[lang];
  safeText("h1", t.h1);
  safeText("p1", t.p1);
  safeText("launchBtn", t.launch);
  safeText("installBtn", t.install);
  safeText("prevTitle", t.prevTitle);
  document.body.dir = (lang === "fa") ? "rtl" : "ltr";
}
$("langToggle")?.addEventListener("click", ()=>{
  lang = (lang === "en") ? "fa" : "en";
  localStorage.setItem("lang", lang);
  applyLang();
});
window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  deferredPrompt = e;
  const btn = $("installBtn");
  if(btn) btn.disabled = false;
});
$("installBtn")?.addEventListener("click", async ()=>{
  if(!deferredPrompt) return;
  deferredPrompt.prompt();
  await deferredPrompt.userChoice;
  deferredPrompt = null;
  if($("installBtn")) $("installBtn").disabled = true;
});
async function loadMeta(){
  try{
    let info = null;
    try{
      const r0 = await fetch("./docs/latest/meta.json", {cache:"no-store"});
      if(r0.ok) info = await r0.json();
    }catch(_){ }
    if(!info){
      const r = await fetch("./docs/latest/meta_index.json", {cache:"no-store"});
      const idx = await r.json();
      const latest = idx.latest_run_id;
      const run = idx.runs.find(x=>x.run_id===latest);
      info = {
        generated_at_utc: run?.generated_at_utc || run?.created_utc || idx.generated_at_utc,
        latest_available_time_id: run?.latest_available_time_id || idx.latest_available_time_id || null,
      };
    }
    const gen = info?.generated_at_utc ? new Date(info.generated_at_utc).toISOString().slice(0,16).replace("T"," ")+" UTC" : "—";
    const lastTid = info?.latest_available_time_id;
    const lastStr = lastTid ? ` • latest data: ${lastTid}` : "";
    safeText("prevMeta", gen + lastStr);
  }catch(e){
    safeText("prevMeta", "—");
  }
}
applyLang();
loadMeta();
