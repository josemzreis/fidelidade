"""Sofia.pt — pipeline: xlsx bruto -> dados limpos -> sinais -> regras -> webapp/data.json + webapp/index.html

Uso:  python pipeline/build_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLSX = ROOT / "Sofia_Hackathon_Dataset_v0_canario.xlsx"
OUT_JSON = ROOT / "webapp" / "data.json"
TEMPLATE = ROOT / "webapp" / "app.template.html"
OUT_HTML = ROOT / "webapp" / "index.html"

START = pd.Timestamp("2026-10-26")
END = pd.Timestamp("2026-11-23")
DAYS = pd.date_range(START, END - pd.Timedelta(days=1), freq="D")

# ---------------------------------------------------------------- load
x = pd.read_excel(XLSX, sheet_name=None)
ben = x["beneficiarios"]; con = x["contactos"]; fic = x["ficha_clinica"]
med = x["medicacao_prescrita"]; cat = x["catalogo_servicos"]; ctr = x["contratos"]
oco = x["ocorrencias"]; rms = x["rm_responsaveis"]; dv = x["dispositivos"]
raw = x["eventos_iot"].copy()

ev = raw.merge(dv[["dispositivo_id", "beneficiario_id", "tipo", "localizacao"]], on="dispositivo_id", how="left")
ev["ts"] = pd.to_datetime(ev.timestamp)
ev["v"] = pd.to_numeric(ev.valor, errors="coerce")
ev = ev.sort_values("ts").reset_index(drop=True)

# ---------------------------------------------------------------- cleaning (Nível 1)
cleaning = []
def drop(mask, rule_id, problem, rule, example=None):
    global ev
    n = int(mask.sum())
    ex = example
    if ex is None and n:
        r = ev[mask].iloc[0]
        ex = f"{r.beneficiario_id} · {r.ts:%d-%b %H:%M} · {r.tipo_evento}={r.valor}"
    cleaning.append(dict(id=rule_id, problema=problem, regra=rule, n=n, exemplo=ex, acao="removido"))
    ev = ev[~mask].reset_index(drop=True)

T = lambda t: ev.tipo_evento == t
drop(ev.duplicated(subset=["timestamp", "dispositivo_id", "tipo_evento", "valor"]),
     "L1", "Evento duplicado com IDs diferentes", "Deduplicar por (timestamp, dispositivo, tipo, valor)")
drop(T("frequencia_cardiaca") & ~ev.v.between(30, 200), "L2", "Frequência cardíaca impossível (0 ou >200 bpm)", "Válido se 30–200 bpm")
drop(T("temperatura") & ~ev.v.between(0, 45), "L3", "Temperatura de sensor em erro (−40 / 85 °C)", "Válido se 0–45 °C (valores-limite = falha do sensor)")
drop(T("humidade") & ~ev.v.between(5, 100), "L4", "Humidade 0 %", "Válido se 5–100 %")
drop(T("spo2") & ~ev.v.between(70, 100), "L5", "SpO₂ fora do possível (0 / 104 %)", "Válido se 70–100 %")
bp = ev[T("tensao_sistolica")][["ts", "dispositivo_id", "v"]].merge(
    ev[T("tensao_diastolica")][["ts", "dispositivo_id", "v"]], on=["ts", "dispositivo_id"], suffixes=("_s", "_d"))
bad_bp = bp[(~bp.v_s.between(70, 250)) | (~bp.v_d.between(40, 140))]
mask_bp = (T("tensao_sistolica") | T("tensao_diastolica")) & ev.ts.isin(bad_bp.ts) & ev.dispositivo_id.isin(bad_bp.dispositivo_id)
drop(mask_bp, "L6", "Par de tensão incoerente (280/38)", "Descartar o par se sistólica ∉ 70–250 ou diastólica ∉ 40–140")
med_peso = ev[T("peso") & (ev.v > 30)].groupby("beneficiario_id").v.median()
dev = (ev.v - ev.beneficiario_id.map(med_peso)).abs() / ev.beneficiario_id.map(med_peso)
drop(T("peso") & ((ev.v <= 30) | (dev > 0.10)), "L7", "Peso impossível (0 kg, 12,4 kg)", "Descartar se desvio >10 % da mediana pessoal")
drop(T("passos") & (ev.v > 2500), "L8", "4100 passos em 15 min (com andarilho)", "Válido se ≤ 2500 passos/15 min")

# flagged-not-removed (o que parece problema mas não é / requer contexto)
ev["h"] = ev.ts.dt.hour
ev["d"] = ev.ts.dt.normalize()
ev["night"] = (ev.h >= 23) | (ev.h < 7)

# ---------------------------------------------------------------- gateway / device health
gw = ev[T("gateway_ligado")]
health = {}
for b, g in gw.groupby("beneficiario_id"):
    gaps = []
    dt = g.ts.diff()
    for i in g.index[dt > pd.Timedelta("90min")]:
        a = g.ts[i] - dt[i]
        gaps.append(dict(inicio=a.strftime("%Y-%m-%d %H:%M"), fim=g.ts[i].strftime("%Y-%m-%d %H:%M"),
                         horas=round(dt[i].total_seconds() / 3600, 1)))
    partial = g[g.v < 20]
    health[b] = dict(uptime=round(len(g) / 672 * 100, 1), gaps=gaps,
                     parcial_horas=int(len(partial)),
                     parcial=(f"{partial.ts.min():%d-%b} → {partial.ts.max():%d-%b}" if len(partial) else None))
bat = ev[T("bateria")]
low = bat[bat.v < 15].groupby(["beneficiario_id", "tipo", "localizacao"]).agg(min=("v", "min"), primeiro=("ts", "min"), ultimo=("ts", "max")).reset_index()
for b in ben.beneficiario_id:
    health.setdefault(b, dict(uptime=100, gaps=[], parcial_horas=0, parcial=None))
    health[b]["baterias"] = [dict(dispositivo=f"{r.tipo} ({r.localizacao})", min=int(r["min"]),
                                  periodo=f"{r.primeiro:%d-%b} → {r.ultimo:%d-%b}") for _, r in low[low.beneficiario_id == b].iterrows()]

# ---------------------------------------------------------------- daily signals
def daily(mask, how="size", col="v"):
    s = ev[mask]
    g = s.groupby(["beneficiario_id", "d"])
    r = g.size() if how == "size" else g[col].agg(how)
    return r

S = {
    "passos": daily(T("passos"), "sum"),
    "pulseira_h": daily(T("frequencia_cardiaca")) / 4,
    "levantes_noite": daily(T("presenca_cama") & (ev.v == 0) & ev.night),
    "wc_noite": daily(T("movimento") & (ev.localizacao == "wc") & (ev.v == 1) & ev.night),
    "doses_falhadas": daily(T("dose_falhada")),
    "doses_tomadas": daily(T("dose_tomada")),
    "temp_interior": daily(T("temperatura"), "mean"),
    "spo2": daily(T("spo2"), "mean"),
    "fc": daily(T("frequencia_cardiaca"), "mean"),
    "sistolica": daily(T("tensao_sistolica"), "mean"),
    "peso": daily(T("peso"), "mean"),
    "frigorifico": daily(T("porta") & (ev.localizacao != "porta_entrada") & (ev.v == 1)),
    "cozinha": daily(T("tomada_ligada") & ev.localizacao.isin(["chaleira", "microondas"])),
    "saidas": daily(T("porta") & (ev.localizacao == "porta_entrada") & (ev.v == 1)),
}
COUNT_METRICS = {"passos", "levantes_noite", "wc_noite", "doses_falhadas", "doses_tomadas", "frigorifico", "cozinha", "saidas", "pulseira_h"}

def series(b, m):
    s = S[m]
    try:
        s = s.xs(b, level=0)
    except KeyError:
        s = pd.Series(dtype=float)
    s = s.reindex(DAYS)
    if m in COUNT_METRICS:
        s = s.fillna(0)
    return [None if pd.isna(v) else round(float(v), 1) for v in s.values]

def wk(arr, w):  # mean of week w (1..4), ignoring None
    vals = [v for v in arr[(w - 1) * 7:w * 7] if v is not None]
    return round(float(np.mean(vals)), 1) if vals else None

# ---------------------------------------------------------------- rules engine
def pct(a, b):
    return None if (a in (None, 0) or b is None) else round((b - a) / a * 100)

def run_rules(b, sig, ficha):
    f = []
    def add(sev, rid, titulo, evid, metric=None):
        f.append(dict(sev=sev, regra=rid, titulo=titulo, evidencia=evid, metric=metric))
    # R1 peso — regra clínica de IC (+2 kg / 7 dias)
    p = [(DAYS[i], v) for i, v in enumerate(sig["peso"]) if v is not None]
    best = None
    for i, (di, vi) in enumerate(p):
        for dj, vj in p[i + 1:]:
            if (dj - di).days <= 7 and vj - vi >= 2.0 and (best is None or vj - vi > best[2]):
                best = (di, dj, vj - vi, vi, vj)
    if best and "Insuficiência cardíaca" in ficha.condicoes_principais:
        add(3, "R1", f"Peso +{best[2]:.1f} kg em {(best[1]-best[0]).days} dias",
            f"{best[3]:.1f} kg ({best[0]:%d-%b}) → {best[4]:.1f} kg ({best[1]:%d-%b}). Indicação médica: avisar se >2 kg/semana. Último: {p[-1][1]:.1f} kg.", "peso")
    elif p and p[-1][1] - p[0][1] <= -1.5:
        add(1, "R1b", f"Perda de peso {p[-1][1]-p[0][1]:.1f} kg em 4 semanas", f"{p[0][1]:.1f} → {p[-1][1]:.1f} kg", "peso")
    # R2 SpO2
    s1, s4 = wk(sig["spo2"], 1), wk(sig["spo2"], 4)
    if s4 is not None and (s4 < 93 or (s1 and s1 - s4 >= 2)):
        mn = min(v for v in sig["spo2"][21:] if v is not None)
        add(2, "R2", f"SpO₂ a descer: {s1}% → {s4}%", f"Média semanal semana 1 vs semana 4; mínimo recente {mn:.0f}%.", "spo2")
    # R3 FC repouso a subir
    f1, f4 = wk(sig["fc"], 1), wk(sig["fc"], 4)
    if f1 and f4 and f4 - f1 >= 5:
        add(2, "R3", f"Frequência cardíaca média +{f4-f1:.0f} bpm", f"{f1} → {f4} bpm (média semanal).", "fc")
    # R4 levantes noturnos
    n1, n4 = wk(sig["levantes_noite"], 1), wk(sig["levantes_noite"], 4)
    if n4 is not None and n4 >= 1.8 and n4 >= 2 * max(n1 or 0, 0.5):
        w1, w4 = wk(sig["wc_noite"], 1), wk(sig["wc_noite"], 4)
        add(2, "R4", f"Levanta-se {n4:.1f}×/noite (era {n1:.1f})", f"Saídas da cama 23h–7h; idas ao WC à noite {w1} → {w4}.", "levantes_noite")
    # R5 medicação
    miss4 = sum(sig["doses_falhadas"][21:])
    run = 0; best_run = 0
    for v in sig["doses_falhadas"]:
        run = run + 1 if v and v > 0 else 0
        best_run = max(best_run, run)
    miss1 = sum(sig["doses_falhadas"][:7])
    if miss4 >= 3 or best_run >= 5:
        add(3 if (best_run >= 7 or miss4 >= 7) else 2, "R5",
            f"{int(miss4)} doses falhadas na última semana", f"Semana 1: {int(miss1)} → semana 4: {int(miss4)}; {best_run} dias seguidos com falha.", "doses_falhadas")
    # R6 temperatura interior
    t4 = wk(sig["temp_interior"], 4); t1 = wk(sig["temp_interior"], 1)
    if t4 is not None and t4 < 16:
        first = next((DAYS[i] for i, v in enumerate(sig["temp_interior"]) if v is not None and v < 16), None)
        add(3 if "DPOC" in ficha.condicoes_principais else 2, "R6", f"Casa a {t4:.1f} °C",
            f"Média interior {t1} → {t4} °C desde {first:%d-%b} (recomendado ≥18 °C).", "temp_interior")
    # R7 atividade (normalizada pelo uso da pulseira)
    a1, a4 = wk(sig["passos"], 1), wk(sig["passos"], 4)
    u1, u4 = wk(sig["pulseira_h"], 1), wk(sig["pulseira_h"], 4)
    if a1 and a4 is not None and pct(a1, a4) <= -40:
        if u1 and u4 and u4 < 0.85 * u1:
            add(0, "R7x", f"Passos {pct(a1,a4)}%, mas a pulseira foi menos usada",
                f"Uso da pulseira {u1:.0f} → {u4:.0f} h/dia: a quebra é do sensor, não da pessoa.", "passos")
        else:
            add(2 if pct(a1, a4) <= -55 else 1, "R7", f"Passos {pct(a1,a4)}% vs semana 1",
                f"{a1:.0f} → {a4:.0f} passos/dia com a pulseira usada {u4:.0f} h/dia.", "passos")
    # R8 cozinha / alimentação
    c1, c4 = wk(sig["cozinha"], 1), wk(sig["cozinha"], 4)
    g1, g4 = wk(sig["frigorifico"], 1), wk(sig["frigorifico"], 4)
    if g1 and g4 is not None and pct(g1, g4) <= -50:
        add(2, "R8", f"Frigorífico {pct(g1,g4)}% · chaleira/micro-ondas {pct(c1,c4)}%",
            f"Aberturas do frigorífico {g1} → {g4}/dia; aparelhos de cozinha {c1} → {c4}/dia.", "frigorifico")
    # R9 tensão
    b1, b4 = wk(sig["sistolica"], 1), wk(sig["sistolica"], 4)
    if b1 and b4 and b4 - b1 >= 10:
        add(1, "R9", f"Tensão sistólica +{b4-b1:.0f} mmHg", f"{b1} → {b4} mmHg (média semanal).", "sistolica")
    return f

# ---------------------------------------------------------------- critical events (pulseira / fumo)
crit = ev[ev.tipo_evento.isin(["queda_detetada", "sos", "alarme_fumo"])]
crit_events = {}
for _, r in crit.iterrows():
    after = ev[(ev.beneficiario_id == r.beneficiario_id) & (ev.ts > r.ts) & (ev.ts <= r.ts + pd.Timedelta("15min"))]
    steps_after = int(after[after.tipo_evento == "passos"].v.sum())
    hr_after = after[after.tipo_evento == "frequencia_cardiaca"].v.tolist()
    crit_events.setdefault(r.beneficiario_id, []).append(dict(
        ts=r.ts.strftime("%Y-%m-%d %H:%M"), tipo=r.tipo_evento, valor=float(r.v) if not pd.isna(r.v) else r.valor,
        passos_15min=steps_after, fc_depois=hr_after[:2]))

# BEN-005 — linha do tempo da queda de 12-nov
fall = ev[(ev.beneficiario_id == "BEN-005") & (ev.ts >= "2026-11-12 15:40") & (ev.ts <= "2026-11-12 18:16") &
          ev.tipo_evento.isin(["queda_detetada", "movimento", "frequencia_cardiaca", "porta", "passos"])]
fall_tl = [dict(t=r.ts.strftime("%H:%M"), tipo=r.tipo_evento, loc=r.localizacao, v=r.valor) for _, r in fall.iterrows()]

# ---------------------------------------------------------------- curated decisions (analista) — regras detetam, pessoa decide
SRV = cat.set_index("servico_id")
def srv(i, qty=None):
    s = SRV.loc[i]
    return dict(id=i, nome=s.nome, preco=float(s.preco_cliente_eur), unidade=s.unidade, qty=qty)

DECISION = {
 "BEN-007": dict(urgencia="hoje", importancia=3, confianca="Alta",
    resumo="Sinais de descompensação da insuficiência cardíaca: o padrão que levou ao internamento de setembro.",
    porque="Quatro sinais independentes apontam no mesmo sentido: peso, oxigénio, pulso e noites (falta de ar deitada).",
    contactos=["CON-011", "CON-012"], canal_override={"CON-012": "Telefone"},
    acao="Ligar hoje à Dra. Inês Valente para reavaliação (ajuste do diurético) e avisar o filho.",
    servicos=[srv("SRV-06", "1 h · consulta")],
    confianca_nota="Balança sem bateria 29-out→02-nov (4 dias sem peso), mas a tendência de 3 semanas é contínua."),
 "BEN-008": dict(urgencia="hoje", importancia=3, confianca="Alta",
    resumo="Casa fria desde 8-nov e, no mesmo dia, deixou de tomar a medicação da manhã. Com DPOC, o frio agrava a respiração.",
    porque="A temperatura caiu 4 °C de um dia para o outro nos dois sensores: avaria da caldeira (já referida na instalação).",
    contactos=["CON-013"], canal_override={"CON-013": "Telefone"},
    acao="Ligar hoje à filha: resolver o aquecimento (aquecedor provisório) e confirmar a medicação no local.",
    servicos=[srv("SRV-04", "diária · 2 semanas")],
    confianca_nota="Gateway desligado 7 h a 18-nov (lacuna, não inatividade). Dispensador a comunicar: as falhas são reais."),
 "BEN-006": dict(urgencia="48h", importancia=3, confianca="Alta",
    resumo="Sinais de abandono do autocuidado 3 meses após ficar viúvo: deixou de comer e de tomar a medicação.",
    porque="Falha doses de metformina e sertralina, quase não abre o frigorífico nem usa a cozinha e perdeu peso. Alarme de fumo a 16-nov.",
    contactos=["CON-009", "CON-010"], canal_override={"CON-009": "Telefone"},
    acao="Ligar à filha e propor reforço do plano (só tem 6 h/semana). Sinalizar ao médico (humor e défice cognitivo).",
    servicos=[srv("SRV-04", "diária"), srv("SRV-03", "3×/sem"), srv("SRV-01", "2 h/sem")],
    confianca_nota="Vários sensores independentes (dispensador, frigorífico, tomadas, balança) concordam."),
 "BEN-005": dict(urgencia="48h", importancia=3, confianca="Alta",
    resumo="Depois da queda de 12-nov quase deixou de andar (−70 % de passos). Risco de nova queda e de perda de autonomia.",
    porque="A pulseira detetou a queda às 15:42 (confiança 0,94), mas ninguém foi avisado: foi encontrada às 18:05.",
    contactos=["CON-008"], canal_override={},
    acao="Falar com a sobrinha, reavaliar o plano e ativar o alerta de queda em tempo real.",
    servicos=[srv("SRV-05", "2 h/sem")],
    confianca_nota="Pulseira usada todo o dia: a quebra de passos é real. Dose das 13:00 (vitamina D) com falhas pontuais."),
 "BEN-004": dict(urgencia="48h", importancia=2, confianca="Média",
    resumo="Levanta-se de noite quase 3× mais. Antes da infeção urinária de março aconteceu o mesmo.",
    porque="Padrão noturno igual ao que antecedeu a infeção. Também pode ser da próstata: só o médico confirma.",
    contactos=["CON-006", "CON-007"], canal_override={},
    acao="Ligar ao filho e marcar médico de família (análise à urina). Mais idas noturnas = mais risco de queda.",
    servicos=[srv("SRV-06", "1 h · consulta")],
    confianca_nota="Sinal forte (cama + WC concordam); a causa é hipótese."),
 "BEN-003": dict(urgencia="semana", importancia=2, confianca="Média",
    resumo="Está a ficar em casa: menos passos, acorda mais tarde e falha o paracetamol. A cuidadora já o notou.",
    porque="A queda de atividade confirma a observação de 11-nov (dores nos joelhos, deixou de ir ao café).",
    contactos=["CON-005"], canal_override={},
    acao="Mensagem à filha; propor estimulação física e consulta para controlo da dor.",
    servicos=[srv("SRV-05", "2 h/sem"), srv("SRV-06", "1 h · consulta")],
    confianca_nota="Tendência gradual; sem sinais agudos. Temperatura de 85 °C a 07-nov era erro do sensor."),
 "BEN-002": dict(urgencia="nenhuma", importancia=1, confianca="Alta",
    resumo="Nada a escalar. Os alertas desta casa são da pulseira, não do António.",
    porque="3 \"quedas\" com confiança 0,58–0,61 seguidas de atividade normal; o SOS de 06-nov teve FC 71 e passos logo a seguir; a pulseira esteve menos usada (bateria a 8 %). Não vive sozinho.",
    contactos=[], canal_override={},
    acao="Operacional: trocar a bateria da pulseira. Confirmar o SOS de 06-nov na próxima chamada de rotina.",
    servicos=[],
    confianca_nota="Um SOS nunca se descarta automaticamente: este foi revisto e é compatível com toque acidental."),
 "BEN-001": dict(urgencia="nenhuma", importancia=1, confianca="Alta",
    resumo="Estável em todos os sinais. As anomalias desta casa são dos sensores.",
    porque="Leituras de −40 °C e 12,4 kg são erros; o sensor de movimento da cozinha ficou sem bateria (05→10-nov).",
    contactos=[], canal_override={},
    acao="Operacional: confirmar a troca da bateria do sensor da cozinha. Nenhum contacto com a família.",
    servicos=[],
    confianca_nota="1 dose falhada (21-nov), isolada."),
}

URG = {"hoje": 4, "48h": 3, "semana": 2, "nenhuma": 0}

# ---------------------------------------------------------------- assemble
out_ben = []
for _, b in ben.iterrows():
    bid = b.beneficiario_id
    ficha = fic.set_index("beneficiario_id").loc[bid]
    sig = {m: series(bid, m) for m in S}
    flags = run_rules(bid, sig, ficha)
    for e in crit_events.get(bid, []):  # R10 eventos críticos da pulseira / fumo
        when = pd.Timestamp(e["ts"]).strftime("%d-%b %H:%M")
        if e["tipo"] == "queda_detetada" and e["valor"] >= 0.8:
            flags.append(dict(sev=3, regra="R10", titulo=f"Queda detetada {when} (confiança {e['valor']:.2f})",
                              evidencia="Pouco movimento e FC 99→112 a seguir. Ninguém foi avisado: encontrada às 18:05 pela cuidadora.", metric="queda"))
        elif e["tipo"] == "queda_detetada":
            flags.append(dict(sev=0, regra="R10x", titulo=f"Queda {when} · confiança {e['valor']:.2f}",
                              evidencia=f"Abaixo de 0,8 e {e['passos_15min']} passos nos 15 min seguintes: falso positivo.", metric=None))
        elif e["tipo"] == "sos":
            flags.append(dict(sev=0, regra="R11", titulo=f"SOS {when} revisto",
                              evidencia=f"FC {e['fc_depois'][0]:.0f} e {e['passos_15min']} passos logo a seguir: provável toque acidental. Confirmar.", metric=None))
        elif e["tipo"] == "alarme_fumo" and e["valor"] == 1:
            flags.append(dict(sev=1, regra="R12", titulo=f"Alarme de fumo {when} (6 min)",
                              evidencia="Reposto com movimento na cozinha. Contexto: utente com défice cognitivo.", metric=None))
    d = DECISION[bid]
    cmap = con.set_index("contacto_id")
    contacts = []
    for cid in d["contactos"]:
        c = cmap.loc[cid]
        contacts.append(dict(id=cid, nome=c.nome, relacao=c.relacao, telefone=c.telefone,
                             email=None if pd.isna(c.email) else c.email, canal_preferido=c.canal_preferido,
                             canal=d["canal_override"].get(cid, c.canal_preferido), prioridade=int(c.prioridade_escalonamento)))
    all_contacts = [dict(nome=r.nome, relacao=r.relacao, canal=r.canal_preferido, prioridade=int(r.prioridade_escalonamento))
                    for _, r in con[con.beneficiario_id == bid].sort_values("prioridade_escalonamento").iterrows()]
    vuln = 0.5 * (b.vive_sozinho == "Sim") + 0.5 * (ficha.risco_queda == "Alto") + 0.25 * (b.grau_dependencia == "Moderada")
    sev = max([f["sev"] for f in flags] or [0])
    c = ctr[ctr.beneficiario_id == bid].iloc[0]
    rm = rms.set_index("rm_id").loc[b.rm_id]
    out_ben.append(dict(
        id=bid, nome=b.nome, idade=int(b.idade), sexo=b.sexo, local=f"{b.freguesia}, {b.concelho}",
        vive_sozinho=b.vive_sozinho == "Sim", dependencia=b.grau_dependencia, rm=dict(id=b.rm_id, nome=rm.nome),
        condicoes=ficha.condicoes_principais, mobilidade=ficha.mobilidade, risco_queda=ficha.risco_queda, notas=ficha.notas_clinicas,
        medicacao=[f"{r.medicamento} · {r.horarios}" for _, r in med[med.beneficiario_id == bid].iterrows()],
        contrato=dict(horas=float(c.horas_semana_contratadas), orcamento=float(c.orcamento_mensal_eur)),
        ocorrencias=[dict(data=r.data, tipo=r.tipo, texto=r.descricao) for _, r in oco[oco.beneficiario_id == bid].iterrows()],
        sinais=sig, semanas={m: [wk(sig[m], w) for w in range(1, 5)] for m in S},
        flags=sorted(flags, key=lambda f: -f["sev"]), eventos_criticos=crit_events.get(bid, []), saude=health[bid],
        decisao={**{k: v for k, v in d.items() if k not in ("contactos", "canal_override")}, "contactos": contacts},
        contactos_todos=all_contacts,
        score=dict(severidade=sev, vulnerabilidade=vuln, urgencia=URG[d["urgencia"]], importancia=d["importancia"] + vuln,
                   total=round(URG[d["urgencia"]] * 10 + sev * 3 + vuln, 2)),
    ))
out_ben.sort(key=lambda r: -r["score"]["total"])

not_problems = [
    dict(caso="Lacunas curtas do gateway (2–3 h)", onde="BEN-001/002/003/005/008", decisao="Ausência de dados ≠ inatividade. Não gera alerta; só gera alerta de sistema se >3 h."),
    dict(caso="Gateway desligado 7 h", onde="BEN-008 · 18-nov 09:00→16:00", decisao="Lacuna técnica. Excluída do cálculo de atividade; aviso técnico, não clínico."),
    dict(caso="Quedas com confiança <0,8 e atividade normal a seguir", onde="BEN-002 (3×), BEN-001 (1×)", decisao="Falsos positivos (tirar/pôr a pulseira). Registados, não escalados."),
    dict(caso="Botão SOS", onde="BEN-002 · 06-nov 10:12", decisao="FC 71 e passos normais a seguir: provável toque acidental. Nunca descartado em automático; confirmar."),
    dict(caso="Alarme de fumo 6 min ao jantar", onde="BEN-006 · 16-nov 19:52", decisao="Cozinhado queimado; reposto com movimento na cozinha. Entra como contexto do caso BEN-006."),
    dict(caso="Queda de passos com pulseira menos usada", onde="BEN-002 · semana 4", decisao="Normalizar passos pelas horas de uso da pulseira antes de concluir."),
    dict(caso="Balança sem bateria (sem pesagens)", onde="BEN-007 · 29-out→02-nov", decisao="Dias sem peso tratados como em falta, nunca como \"peso estável\"."),
    dict(caso="Sensor de movimento sem bateria", onde="BEN-001 · cozinha 05→10-nov", decisao="Gateway reportou 19/20 dispositivos. Menos movimento na cozinha nesse período não é sinal."),
]

data = dict(
    gerado=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    periodo=dict(inicio="2026-10-26", fim="2026-11-22", semana_atual="16–22 nov"),
    dias=[d.strftime("%Y-%m-%d") for d in DAYS],
    totais=dict(eventos_brutos=int(len(raw)), eventos_limpos=int(len(ev)), removidos=int(len(raw) - len(ev)), dispositivos=int(len(dv))),
    limpeza=cleaning, nao_problemas=not_problems,
    beneficiarios=out_ben, queda_005=fall_tl,
    rms=[dict(id=r.rm_id, nome=r.nome) for _, r in rms.iterrows()],
    catalogo=[dict(id=r.servico_id, nome=r.nome, preco=float(r.preco_cliente_eur), unidade=r.unidade) for _, r in cat.iterrows()],
)

def clean(o):
    if isinstance(o, dict): return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list): return [clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, float) and np.isnan(o): return None
    return o

data = clean(data)
OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
if TEMPLATE.exists():
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    # fragmento (pré-visualização) + documento completo para a Vercel
    (ROOT / "webapp" / ".preview.html").write_text(html, encoding="utf-8")
    full = ('<!doctype html>\n<html lang="pt-PT">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
            + html + '\n</body>\n</html>\n')
    OUT_HTML.write_text(full, encoding="utf-8")
print(f"ok · {len(raw)} → {len(ev)} eventos · {OUT_JSON.stat().st_size//1024} KB")
for b in out_ben:
    print(b["id"], b["decisao"]["urgencia"], b["score"]["total"], [f"{f['regra']}:{f['titulo']}" for f in b["flags"]])
