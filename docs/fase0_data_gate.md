# Fase 0 — Data Gate: Optimizador Multi-Perfil HRP sobre FICs (SFC / datos.gov.co)

**Fecha de auditoría:** 2026-09-11 · **Fuente:** API SODA de datos.gov.co, sin app token · **Estado del gate:** ✅ APROBADO por Carlos el 2026-09-11 — Opción 1 en D1–D11 (ver §5)
**Evidencia reproducible:** `docs/fase0_evidence/scripts/` (14 scripts Python) y `docs/fase0_evidence/data/` (salidas CSV). Todo número marcado [VERIFICADO] sale de una query listada en el Anexo A.

Convención de etiquetas:
- **[VERIFICADO]** — medido directamente contra la API, con la query como evidencia.
- **[INFERIDO]** — deducción razonable a partir de datos verificados o de una norma citada, no confirmada por una fuente directa.
- **[ESPECULATIVO]** — supuesto de diseño sin fuente; requiere decisión explícita.

---

## 0. Resumen ejecutivo y veredicto del gate

**Veredicto: NO hay bloqueante.** La profundidad histórica real es **10.7 años (2016-01-01 → 2026-09-08)** [VERIFICADO], muy por encima del umbral de 2 años. Sin embargo, hay **tres desviaciones respecto al diseño asumido en la investigación previa** que cambian la arquitectura y requieren decisión antes de la Fase 1:

| # | Hallazgo | Impacto en diseño | Etiqueta |
|---|---|---|---|
| H1 | **Dataset B (`djw7-ur7t`, "Filtro_Fondo") NO es una tabla de metadata/categorización.** Tiene el mismo esquema de 26 columnas que A y es un **subconjunto estricto** de A (solo 2024-01-01 en adelante; 980,944 filas; conteos por año idénticos a A). El único campo de categoría es `nombre_subtipo_patrimonio` con 5 valores regulatorios (General / Mercado Monetario / Inmobiliarias / Bursátiles / FCP) — no distingue RF corto vs. largo, ni RV local vs. internacional, ni mixtos. | La clasificación en 5 categorías **debe derivarse** (nombre + volatilidad realizada + correlación con proxies). Ver §2.1. | [VERIFICADO] |
| H2 | **Reasignación de códigos CONFIRMADA.** Cuando un fondo cambia de administrador (`codigo_entidad`) — y ocasionalmente sin cambiarlo — la SFC emite un `codigo_negocio` nuevo. Detectados **71 traspasos** con el mismo nombre y brecha de 1 día (66 con cambio de entidad; 48 de ellos el 2026-01-01 por la consolidación en Fiduaval). Afecta a fondos del universo (p. ej. ETF Global X Colombia Select: `43502` → `129433` el 2026-01-01). | El identificador de fondo para el backtest **no puede ser `codigo_negocio`**; hay que construir una tabla maestra que encadene códigos (por nombre normalizado + continuidad del valor de unidad). | [VERIFICADO] |
| H3 | **La categoría "RV internacional" es delgada antes de 2021.** Solo 1 fondo limpio vivo con historia desde 2016 (Credicorp Acciones Globales) más 2 híbridos; los fondos globales "puros" (p. ej. Renta Sostenible Global) nacen en 2021. "Mixtos" tiene 2–3 fondos desde ≤2018. | Determina el trade-off de ventana (§4): profundidad vs. amplitud del universo. | [VERIFICADO] |

Además, dos correcciones de supuestos: (a) la frecuencia es **diaria calendario (365 días/año, incluye fines de semana y festivos)**, no diaria bursátil; (b) los campos `rentabilidad_*` son **tasas efectivas anuales en %**, no retornos simples — el retorno para el modelo debe derivarse de `valor_unidad_operaciones`.

---

## 1. TAREA 1 — Auditoría de los datasets SFC

### 1.0 Cliente, método y rate limit

- Cliente: `sodapy 2.2.0` (`Socrata("www.datos.gov.co", None)`) para metadata; `requests` directo sobre `https://www.datos.gov.co/resource/<id>.json` con `$select/$where/$group/$limit/$offset` para las queries (mismo endpoint SODA 2.1; se usó `requests` por control de timeouts). Python 3.14, pandas 3.0.3.
- **Rate limit observado sin app token [VERIFICADO]:**
  - 160 requests ligeras (`$limit≤100`) en secuencia + ráfaga de 60 requests con 20 hilos: **0 respuestas 429**, sin cabecera `Retry-After`; latencia media 0.9–1.0 s, máx 2.9 s.
  - El throttling se manifiesta en **queries agregadas pesadas**: `count(distinct …)` sobre 2.9 M filas tardó 37 s; un `$group` de 5 columnas con `min/max` sobre toda la tabla fue abortado por el servidor (`Connection reset by peer`) tras >10 min, y una request concurrente ligera sufrió `Read timed out (60 s)` mientras esa query estaba en vuelo. Particionando por año (`$where fecha_corte between …`) las mismas agregaciones respondieron en 1–3 s cada una.
  - Documentación Socrata: sin token, "IP addresses that make too many requests during a given period may be subject to throttling"; con token no se throttlea salvo abuso ([dev.socrata.com/docs/app-tokens.html](https://dev.socrata.com/docs/app-tokens.html)). **Recomendación:** registrar app token (gratuito) antes de la Fase 1 y particionar las descargas por año/fondo. [INFERIDO]
- Lag de publicación [VERIFICADO]: `max(fecha_corte)` = 2026-09-08 consultado el 2026-09-11; cabecera `X-SODA2-Truth-Last-Modified: Thu, 10 Sep 2026`. Lag ≈ 2–3 días calendario.

### 1.1 Dataset A — `qhpu-8ixx` "Rentabilidades de los FIC"

| Ítem | Resultado | Etiqueta | Evidencia |
|---|---|---|---|
| Filas totales | **2,918,294** | [VERIFICADO] | Q1 |
| Rango de fechas | **2016-01-01 → 2026-09-08** (3,904 fechas distintas = 3,904 días calendario; **0 fechas faltantes**) | [VERIFICADO] | Q1, Q4 |
| Frecuencia real | **Diaria calendario.** Cada año tiene 365/366 fechas; 104–105 fechas de fin de semana por año; filas por fecha uniformes entre lunes y domingo (mediana 717–720). Los fondos de mercado monetario devengan sábados y domingos (retorno ≠ 0); los fondos de RV repiten el valor de unidad (retorno = 0). Los festivos colombianos aparecen como días hábiles con retorno 0 en RV (15–18 por año en el ETF COLCAP). | [VERIFICADO] | Q4, Q6 |
| Clave de fila | `(fecha_corte, codigo_negocio, tipo_participacion)`. Un `codigo_negocio` tiene 1–15 clases (`tipo_participacion`); `valor_fondo_cierre_dia_t` es **por clase** (las clases de Fidurenta suman el AUM del fondo). En el corte 2026-09-08: 1,037 filas, 447 códigos, 1,031 claves únicas → 6 filas duplicadas exactas (3 fondos). | [VERIFICADO] | Q5 |
| Códigos únicos | **1,013 `codigo_negocio` distintos** en 2016–2026 (por subtipo: 570 General, 16 Mercado Monetario, 28 Inmobiliarias, 7 Bursátiles, 401 FCP — algunos códigos cambian de subtipo, de ahí 1,022 combinaciones). Por año: 420 (2016) → 483 (2026). **Vivos a 2026-09: 448; muertos: 565** (72 en 2016, 67 en 2017, … 72 en 2025). | [VERIFICADO] | Q2, Q3, Q7b |
| Reasignación de código | **CONFIRMADA.** 71 pares (código A termina el día *t*, código B con el mismo nombre normalizado empieza en *t+1*; 69 con brecha de exactamente 1 día). **66 de 71 coinciden con cambio de `codigo_entidad`** (cambio/fusión de administrador): 48 ocurren el 2026-01-01 hacia Fiduaval (ent. 20) — 36 desde ent. 21, 9 desde ent. 22 (incl. ETFs Global X), 3 desde ent. 18 —, 9 hacia Progresión SCB (ent. 9) en 2022-06/08, 4 desde Interbolsa (ent. 68) el 2024-09-07, 4 de ent. 23→88 el 2024-03-01. Los 5 restantes son recodificaciones dentro de la misma entidad (2017 y 2022). Continuidad del valor de unidad verificada en los 5 traspasos del universo candidato (p. ej. Global X Colombia Select: VU 22,826.99 el 2025-12-31 bajo `43502` y 22,826.99 el 2026-01-01 bajo `129433`). **Nunca** se observó el mismo código bajo 2 entidades (0 casos) y solo 2 códigos cambian de nombre. Riesgo adicional: en el traspaso las clases pueden renumerarse (caso Occitesoros: clase 800 existe solo en el código antiguo). | [VERIFICADO] | Q7b, Q8 |
| Unidades de los campos | `valor_unidad_operaciones` = **valor de la unidad (NAV por participación, COP)**, no NAV total. `valor_fondo_cierre_dia_t` = **valor total del fondo por clase (COP)** = proxy de AUM. `rentabilidad_diaria` = ((VU_t/VU_{t−1})^365 − 1)·100 → **retorno de 1 día anualizado como tasa efectiva anual (E.A.) en %** (coincide a 6 decimales). `rentabilidad_mensual` = ((VU_t/VU_{t−1 mes})^(365/d) − 1)·100 (mismo día del mes anterior, E.A. %). `rentabilidad_semestral` = ((VU_t/VU_{t−6 meses})^(365/d) − 1)·100 (E.A. %). `rentabilidad_anual` = (VU_t/VU_{t−365d} − 1)·100 (simple, %). | [VERIFICADO] | Q6 |
| Calidad de `rentabilidad_diaria` | Rango declarado −5.4e30 … 9.8e31: el exponente 365 amplifica movimientos diarios grandes (un +2.08 % diario del ETF COLCAP se reporta como 183,446 % E.A.). **No usar este campo como retorno**; derivar retornos del valor de unidad. | [VERIFICADO] | Q6 |
| Completitud (muestra n = 47 fondos candidatos, todas las clases descargadas, 490 k filas) | % de días calendario faltantes entre primera y última fecha: **mediana 0.00 %, media 0.16 %, máx 4.01 %**. 43/47 fondos con brecha máxima ≤ 3 días. Excepciones: Alianza Cash 1525 (111 días, 2019), Fiduprevisora Alta Liquidez (31 días), Ashmore Acciones Col+LATAM (29 días), Itaú Acciones Colombia (6 brechas ≤5 días). 0 valores de unidad ≤ 0 en la muestra. Duplicados: 1 fondo (Renta Acciones LATAM, `3644`) con **7,802 filas triplicadas exactas** (3 copias de cada fila) — deduplicación trivial por clave. | [VERIFICADO] | Q8, Q9 |
| Datos extremos reales | Salto diario > 15 % en ETF COLCAP: solo 2020-03-16 (−16.0 %, crash COVID; real, no error). | [VERIFICADO] | Q9 |

### 1.2 Dataset B — `djw7-ur7t` "Filtro_Fondo"

| Ítem | Resultado | Etiqueta | Evidencia |
|---|---|---|---|
| Esquema | **26 columnas idénticas a A** (mismos `fieldName`, mismos tipos). Misma descripción textual, misma dependencia (Delegatura de Fiduciarias). | [VERIFICADO] | Q0 (metadata) |
| Rango de fechas | **2024-01-01 → 2026-09-08**, 982 fechas distintas, 980,944 filas. | [VERIFICADO] | Q1 |
| Relación con A | Subconjunto estricto: filas por año 344,607 / 367,978 / 268,359 en ambos; 617 códigos, 562 nombres, 32 entidades. | [VERIFICADO] | Q3 |
| Campos de categorización | Ninguno adicional. `nombre_subtipo_patrimonio` ∈ {FIC DE TIPO GENERAL, FIC DE MERCADO MONETARIO, FIC INMOBILIARIAS, FIC BURSATILES, FONDOS DE CAPITAL PRIVADO}. Sin campo de política de inversión, benchmark, moneda, duración ni perfil de riesgo. | [VERIFICADO] | Q2 |
| Uso recomendado | Espejo liviano (1/3 del tamaño) para queries recientes; **no aporta información que A no tenga**. El supuesto de la documentación previa ("dataset de categorización") es incorrecto. | [INFERIDO] | — |

### 1.3 Implicaciones de ingeniería de datos (para Fase 1, no implementar aún)

1. Deduplicar por `(fecha_corte, codigo_negocio, tipo_participacion)`.
2. Construir **tabla maestra de fondos**: `fund_id` ← nombre normalizado; encadenar `codigo_negocio` cuando A termina en *t* y B comienza en *t+1* con VU continuo (tolerancia ±2 % diario); elegir por segmento la clase con más historia/AUM y verificar continuidad en el empalme.
3. Serie de precios = `valor_unidad_operaciones` de la clase principal; retornos log sobre calendario **hábil colombiano** (excluir fines de semana y festivos vía `holidays.CO`), de modo que el devengo del fin de semana de los fondos monetarios cae en el retorno del lunes (comportamiento real del inversionista).
4. Excluir subtipo 7 (FCP), subtipo 3 (inmobiliarias: valoración por avalúo, ventanas de redención) y fondos cerrados con < 50 inversionistas.

---

## 2. TAREA 2 — Universo curado (15–20 FICs)

### 2.1 Método de clasificación (sustituye la taxonomía inexistente de B)

Para los **315 fondos FIC (no FCP) con ≥ 24 meses de historia** (171 vivos), se calculó con valores de unidad de inicio de mes (Q10, 68,185 filas): volatilidad anualizada, correlación con tres proxies — **iShares MSCI COLCAP** (RV local), **Credicorp Acciones Globales** (RV global) y **Fiducuenta** (mercado monetario) — y max drawdown. Reglas [INFERIDO, umbrales de diseño]:

| Clase derivada | Regla | Vivos con historia ≤ 2018-06 |
|---|---|---|
| RF corto / mercado monetario | vol < 2 % | 44 |
| RF medio-largo | 2 % ≤ vol < 8 % y corr(COLCAP) < 0.5 y corr(Global) < 0.5 | 15 (de los cuales 4 son inmobiliarios → excluidos) |
| RV local | corr(COLCAP) ≥ 0.8 | 10 |
| RV internacional / global | corr(Global) ≥ 0.5 y corr(COLCAP) < 0.6 | 3 |
| Mixto | resto con vol < 8 % | 3 |
| RV / alternativo otro | vol ≥ 8 % sin encajar arriba | 4 |

La regla por nombre (regex) **falla** en casos relevantes y se usó solo como pista: "ACCION UNO" es monetario (vol 0.6 %), "SURA MULTIESTRATEGIA CR…" es crédito (vol 0.7 %), "RENTA ALTA CONVICCIÓN" es global de alta vol (13.9 %, corr Global 0.70), "FIDUGOB" y "CXC" se comportan como monetarios (vol 0.6 % y 0.3 %). [VERIFICADO]

### 2.2 Criterio de ranking (explícito, en orden)

1. **Filtro duro:** vivo a 2026-09-08; subtipo ∉ {FCP, Inmobiliarias}; abierto o cerrado con ≥ 50 inversionistas; completitud ≥ 99 % y brecha máxima ≤ 31 días.
2. **Pureza de categoría:** cumple la regla de §2.1 (corr/vol) — no solo el nombre.
3. **Profundidad:** inicio ≤ 2018-01 → `core`; inicio 2018-02…2021-06 → `late` (entra al universo cuando acumule la ventana de estimación, punto-en-el-tiempo).
4. **Tamaño / liquidez:** AUM del fondo (suma de clases) a 2026-09-08, descendente; mínimo 15 mil M COP para `core`.
5. **Diversificación de administrador:** máximo 2 fondos del mismo grupo por categoría.

### 2.3 Universo propuesto

Métricas sobre retornos mensuales 2016-01 → 2026-09 (o desde inicio); `vol`/`ret` anualizadas en %, `mdd` = max drawdown mensual %, `c_*` = correlación con COLCAP / Global / Fiducuenta; AUM en miles de millones de COP. [VERIFICADO] (Q10, Q11)

| Cat | Rol | Códigos (encadenados) | Clase | Fondo | Administrador | Inicio | AUM | vol | ret | mdd | c_colcap | c_glob | c_mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RF_CORTO | core | 2852 | 800 | FIC Abierto Fiducuenta | Fiduciaria Bancolombia | 2016-01 | 25,777 | 1.3 | 6.5 | −0.3 | 0.02 | 0.00 | 1.00 |
| RF_CORTO | core | 11407 | 518 | Cartera Colectiva Abierta de Alta Liquidez (subtipo Mercado Monetario) | Fiduprevisora | 2018-02 | 7,622 | 1.2 | 6.6 | −0.2 | 0.01 | 0.00 | 0.98 |
| RF_CORTO | core | 58756 | 800 | Credicorp Capital Alta Liquidez | Credicorp Capital | 2016-01 | 7,277 | 1.1 | 6.1 | −0.5 | 0.11 | 0.07 | 0.96 |
| RF_CORTO | core | 8734 | 501 | FIC Abierto BBVA Efectivo | BBVA AM | 2016-01 | 6,140 | 1.0 | 6.3 | −0.1 | 0.06 | 0.05 | 0.97 |
| RF_LARGO | core | 3049 | 800 | FIC Abierto CPP Plan Semilla | Fiduciaria Bancolombia | 2016-01 | 707 | 2.7 | 7.4 | −4.3 | 0.20 | 0.16 | 0.77 |
| RF_LARGO | core | 58883 | 800 | Credicorp Capital Deuda Corporativa | Credicorp Capital | 2016-01 | 522 | 2.4 | 7.0 | −2.1 | 0.18 | 0.18 | 0.77 |
| RF_LARGO | core | 59076 | 501 | FIC Renta Fija Mediano Plazo Multiescala | Corredores Davivienda | 2017-07 | 259 | 2.7 | 6.8 | −3.5 | 0.20 | 0.17 | 0.76 |
| RF_LARGO | core | 61583 | 800 | Universitas FIC Abierto CPP | Fiduciaria Caja Social | 2016-01 | 151 | 3.2 | 5.5 | −5.4 | 0.22 | 0.15 | 0.68 |
| RF_LARGO | late | 78886 | 501 | FIC Renta Fija Largo Plazo | Corredores Davivienda | 2019-02 | 93 | 4.9 | 6.4 | −12.6 | 0.29 | 0.28 | 0.61 |
| RV_LOCAL | core | 60678 | 800 | Fondo Bursátil iShares MSCI COLCAP | Citivalores | 2016-01 | 9,869 | 20.6 | 7.2 | −35.5 | 1.00 | 0.39 | 0.02 |
| RV_LOCAL | core | 43502 → 129433 | 800 | Fondo Bursátil Global X Colombia Select de S&P | Fiduaval (antes ent. 22) | 2016-01 | 3,077 | 21.0 | 9.5 | −36.5 | 0.98 | 0.42 | 0.04 |
| RV_LOCAL | core | 59105 | 800 | Credicorp Capital Acciones Colombia | Credicorp Capital | 2016-01 | 406 | 20.0 | 13.9 | −32.9 | 0.97 | 0.40 | 0.05 |
| RV_LOCAL | core | 58735 | 800 | FIC Abierto Accival Acciones Dinámico | Acciones y Valores | 2016-01 | 159 | 18.9 | 10.4 | −30.8 | 0.98 | 0.40 | 0.04 |
| RV_INTL | core | 58918 | 800 | Credicorp Capital Acciones Globales | Credicorp Capital | 2016-01 | 63 | 14.5 | 8.6 | −23.0 | 0.39 | 1.00 | 0.00 |
| RV_INTL | core | 59079 | 501 | FIC Acciones Globales ⚠ híbrido (c_colcap 0.74) | Corredores Davivienda | 2016-01 | 19 | 18.1 | 1.2 | −36.7 | 0.74 | 0.67 | 0.06 |
| RV_INTL | core | 78888 | 501 | FIC Diversificado Internacional ⚠ multiactivo global (vol 9 %) | Corredores Davivienda | 2018-11 | 17 | 9.0 | 5.3 | −15.9 | 0.08 | 0.69 | −0.04 |
| RV_INTL | late | 59740 | 803 | FIC Abierto Renta Sostenible Global (el más "puro": c_glob 0.94) | Valores Bancolombia | 2021-03 | 176 | 11.9 | 2.1 | −17.4 | −0.19 | 0.94 | −0.07 |
| MIXTO | core | 3078 | 800 | FIC Abierto Renta Balanceado | Fiduciaria Bancolombia | 2016-01 | 275 | 6.9 | 7.7 | −13.5 | 0.71 | 0.78 | 0.23 |
| MIXTO | core | 69368 | 800 | FIC Abierto CPP BBVA AM Estrategia Balanceado Global | BBVA AM | 2017-04 | 57 | 11.2 | 9.8 | −19.6 | 0.80 | 0.82 | 0.07 |
| MIXTO | core | 59078 | 501 | FIC Diversificado Moderado | Corredores Davivienda | 2017-07 | 48 | 6.9 | 5.3 | −13.5 | 0.67 | 0.71 | 0.21 |
| MIXTO | late | 93223 | 800 | Credicorp Capital Balanceado Colombia | Credicorp Capital | 2020-06 | 33 | 5.1 | 7.2 | −9.5 | 0.62 | 0.54 | 0.30 |

**Totales:** 18 `core` (disponibles desde ≤ 2018-01 salvo 78888) + 3 `late` = **21**; el universo activo en cada fecha de rebalanceo es el que cumple la ventana de estimación (17 en 2020-01, 18 desde 2020-11, 21 desde 2023-04).

**Alternos (mismo criterio, para sustitución si Carlos objeta alguno):** RF_CORTO: Fondo Abierto Alianza (10936; 9,782), Sumar (22969→129120; 6,417). RF_LARGO: Skandia Multiplazo (51959), Daviplus Renta Fija Pesos (9792). RV_LOCAL: Fondo de Seguridad Bolívar (65875, 2016-11), Alianza Acciones (67403, 2017-10). RV_INTL: Renta Acciones LATAM (3644, 2016-01; c_colcap 0.87 → en la práctica es Andino), Credicorp Acciones LATAM (99106, 2021-06). Detalle en `docs/fase0_evidence/data/universe_proposed.csv`.

### 2.4 Advertencias sobre el universo

- **Sesgo de supervivencia [VERIFICADO]:** 565 códigos murieron en el periodo; el universo solo contiene fondos vivos hoy (requisito de invertibilidad). Dos fondos de RF largo con vol ≥ 3.4 % murieron en 2024 (Renta Fija Plazo Bancolombia `3814` y Credicorp Renta Fija Colombia `58897`). Se recomienda un test de robustez incluyéndolos con salida punto-en-el-tiempo (§4.3, test "R").
- **"RF largo plazo" en el mercado colombiano de FICs abiertos es en realidad "mediano plazo"** (vol 2.4–3.2 %); solo `78886` (desde 2019) tiene duración larga (vol 4.9 %). El ETF Global X TES (`118898→131893`) sería ideal pero nace en 2024-03. [VERIFICADO]
- **Homogeneidad RF corto [VERIFICADO]:** corr entre monetarios 0.95–0.99. HRP los agrupará en un clúster; la restricción por fondo (§3) evita que uno solo se lleve todo el bloque.
- **Pactos de permanencia [INFERIDO por nombre]:** Plan Semilla, Universitas, BBVA Estrategia Balanceado Global (y alternos Alianza Acciones, Sumar) tienen "con pacto de permanencia" → penalidades por retiro anticipado; favorece rebalanceo trimestral (§4).
- **Concentración por administrador [VERIFICADO]:** Credicorp 5, Davivienda (Corredores) 5, Bancolombia (Fiduciaria + Valores) 4, BBVA 2, y 1 cada uno Fiduprevisora, Caja Social, Citivalores, Fiduaval, Acciones y Valores. Máximo 24 % del universo en un solo grupo.

---

## 3. TAREA 3 — Bandas de restricción por perfil

### 3.1 Evidencia disponible

- **[VERIFICADO — norma]** Decreto 2555 de 2010 (modificado por Decreto 2955 de 2010), régimen de inversión de los fondos de pensiones obligatorias (multifondos), **Art. 2.6.12.1.4**: límites máximos en títulos y/o valores participativos — *Fondo Conservador hasta 20 %, Fondo Moderado hasta 45 %, Fondo de Mayor Riesgo hasta 70 % del valor del fondo*; el límite mínimo del Mayor Riesgo no puede ser inferior al máximo del Moderado (45 %) y el mínimo del Moderado no puede ser inferior al máximo del Conservador (20 %). **Art. 2.6.12.1.10**: exposición a un mismo emisor ≤ 10 % del fondo. **Art. 2.6.12.1.5 num. 14**: Conservador, moneda extranjera sin cobertura ≤ 10 %. Fuente: [Decreto 2955 de 2010 — Función Pública](https://www.funcionpublica.gov.co/eva/gestornormativo/norma.php?i=40123) (texto verificado por grep en la auditoría).
- **[ESPECULATIVO]** No encontré un estándar público numérico para perfiles de *retail wealth management* en Colombia (la SFC exige perfilamiento del inversionista pero deja los umbrales a cada entidad). Las bandas de multifondos son la referencia pública más cercana y se usan aquí **por analogía**, no por obligación regulatoria (este optimizador no es una AFP).

### 3.2 Propuesta de tabla de bandas

Convención: RV combinada = RV local + RV internacional + **50 % de cada fondo mixto** (look-through) [ESPECULATIVO — alternativa: usar la beta del mixto vs. COLCAP/Global estimada en la ventana]. RF = RF corto + RF largo + 50 % mixtos.

| Perfil | Máx RV combinada | Mín RF | Máx RV internacional | Vol anualizada esperada (validación ex-post, no target del optimizador) | Máx por fondo | Mín por fondo activo | Etiqueta |
|---|---|---|---|---|---|---|---|
| Conservador | **20 %** | **70 %** | 10 % | 3–5 % | 25 % | 0 % (o 2 % si activo) | Límites RV/RF: [INFERIDO] de Art. 2.6.12.1.4; FX 10 %: [INFERIDO] de Art. 2.6.12.1.5; vol: [INFERIDO] de vol realizada del universo (RV ≈ 20 %, RF ≈ 1.3 %, corr ≈ 0.05 ⇒ σ ≈ 4.2 %); máx por fondo: [ESPECULATIVO] |
| Moderado | **45 %** | **40 %** | 20 % | 7–10 % (σ implícita ≈ 9.1 %) | 20 % | 0 % (o 2 %) | ídem; 20 % por fondo: [ESPECULATIVO] (Art. 2.6.12.1.10 daría 10 %, demasiado restrictivo con 4 fondos por categoría) |
| Agresivo | **70 %** | **20 %** | 35 % | 11–15 % (σ implícita ≈ 14.0 %) | 20 % | 0 % (o 2 %) | ídem |

Notas de diseño:
- Con HRP, las bandas se aplican como **post-proceso** (proyección de los pesos HRP al poliedro de restricciones, p. ej. `cvxpy` minimizando ‖w − w_HRP‖² sujeto a las bandas) o como **HRP jerárquico por categoría** (HRP intra-categoría + pesos de categoría fijados por perfil). Decisión D5 abajo. [INFERIDO]
- El "target de volatilidad" **no aplica** como entrada de HRP; se propone como banda de validación ex-post: si la vol realizada OOS del perfil cae fuera de la banda ±2 pp, se revisan los límites. [INFERIDO]
- Mínimos por fondo: 0 % permite que HRP+restricciones apague fondos; 2 % evita "polvo" operativo. [ESPECULATIVO]

---

## 4. TAREA 4 — Decisión de ventana de backtest

### 4.1 Hechos que condicionan la ventana [VERIFICADO]

- Fin de datos: 2026-09-08 → último mes completo **2026-08**. Meses disponibles desde 2016-01: **128**.
- Fondos `core` con datos desde 2016-01: 13 (RF_CORTO 3, RF_LARGO 3, RV_LOCAL 4, RV_INTL 2, MIXTO 1). Desde ≤ 2018-01: 17 (se suman Fiduprevisora, Multiescala, Diversificado Moderado, BBVA Estrategia Global; 78888 entra 2018-11). Desde ≤ 2021-06: 21.
- Regímenes cubiertos: 2016 (post-choque petrolero, TES altos), 2018 (elecciones, EM sell-off), **2020 (COVID: COLCAP −16 % en un día; MDD RV local −35 %)**, **2022 (choque de tasas: RF largo MDD −12.6 %)**, 2023–2025 (rally COLCAP +40 % anual). Un backtest que empiece en 2021 solo ve el choque de 2022.
- Observaciones por ventana de estimación con retornos **diarios hábiles** (≈ 245/año tras excluir festivos): 12 m ≈ 245, 24 m ≈ 490, 36 m ≈ 735 para N ≤ 21 activos (ratio T/N ≥ 12 en todos los casos; HRP no invierte la matriz, pero la estructura de correlación sí la usa). Con retornos mensuales, 24 m = 24 obs para N = 21 → **inaceptable**; con semanales 24 m ≈ 104 obs → aceptable.

### 4.2 Opciones y puntos de rebalanceo OOS resultantes

Puntos OOS = meses entre (inicio de estimación + ventana) y 2026-08, ÷ 1 (mensual) o ÷ 3 (trimestral).

| Opción | Inicio estimación | Ventana | Primer rebalanceo OOS | Fondos disponibles en el 1er OOS | OOS mensual | OOS trimestral | Regímenes OOS | Comentario |
|---|---|---|---|---|---|---|---|---|
| **A** "Máxima profundidad" | 2016-01 | rolling 24 m | 2018-01 | 13 | 104 | 34 | 2018 EM, COVID, 2022, rally | RV_INTL = 2 (uno híbrido), MIXTO = 1 → perfiles agresivo/moderado poco diversificados hasta 2019–2020 |
| **B** "Equilibrada" ✅ recomendada | 2018-01 | rolling 24 m | **2020-01** | **17** (78888 entra 2020-11) | **80** | **26** | **COVID (1er rebalanceo justo antes del crash), 2022, rally** | Todas las categorías con ≥ 3 fondos desde el inicio; `late` entran punto-en-el-tiempo (21 fondos desde 2023-04) |
| B' | 2018-01 | rolling 36 m | 2021-01 | 17 | 68 | 22 | 2022, rally (pierde COVID OOS) | Estimación más estable, menos regímenes OOS |
| B'' | 2018-01 | rolling 12 m | 2019-01 | 17 | 92 | 30 | + 2019 | Más reactivo, más turnover, correlaciones ruidosas |
| **C** "Universo completo" | 2021-07 | rolling 24 m | 2023-07 | 21 | 38 | 12 | solo rally 2023–25 | 12 puntos trimestrales ⇒ sin poder estadístico (Harvey & Liu 2015); no ve ninguna crisis OOS. **No recomendada** |
| **E** Expanding | 2018-01 | expanding (mín. 24 m) | 2020-01 | 17 | 80 | 26 | igual que B | Pesos más estables; ignora cambios de régimen (2022 pesaría poco). Usar como *sensibilidad*, no como principal |

### 4.3 Recomendación [INFERIDO]

- **Ventana:** rolling **24 meses (≈ 490 días hábiles)** con retornos diarios hábiles y estimador de covarianza **Ledoit-Wolf** (para la matriz de distancias de HRP); sensibilidad con 12 m y 36 m y con *expanding*.
- **Rebalanceo:** **trimestral** como caso base (26 puntos OOS; compatible con pactos de permanencia y con el costo operativo de mover FICs), **mensual como sensibilidad** (80 puntos). Semanal descartado (costos/pactos).
- **Universo dinámico punto-en-el-tiempo:** un fondo entra cuando acumula 24 m de historia; sale si deja de reportar (evita look-ahead en la composición).
- **Test de robustez "R" (opcional):** incluir `3814` y `58897` (muertos en 2024) con salida punto-en-el-tiempo para acotar el sesgo de supervivencia.
- **Retorno diario:** log(VU_t/VU_{t−1}) sobre calendario hábil colombiano (`holidays.CO`), ffill del valor de unidad en festivos; anualización con 252 (o 245 efectivo). Costos: 0 comisiones explícitas de entrada/salida en FICs abiertos (las comisiones de administración ya están en el VU) + penalidad de pacto de permanencia parametrizable [ESPECULATIVO, verificar reglamentos de los 5 fondos CPP en Fase 1].

---

## 5. Tabla de decisiones propuestas (aprobación explícita)

Marcar una opción por fila (o proponer "Otra"). Sin aprobación no se inicia la Fase 1.

> **Aprobado 2026-09-11 (Carlos, en chat):** Opción 1 en todas las filas. Decisiones adicionales tomadas al aprobar: (a) la matriz de retornos cubre 2016-01-01 → última fecha (downstream recorta a 2018-01 para la Opción B); (b) brechas intra-vida → NaN en los días sin dato y retorno acumulado en el día de reanudación, registrado en `manifest.json → gap_log`. Implementación: `data/ingest_sfc.py` + `data/universe.json` (ver `data/README.md`).

| ID | Decisión | Opción 1 (recomendada) | Opción 2 | Opción 3 | Tu elección |
|---|---|---|---|---|---|
| **D1** | Identificador de fondo | **Tabla maestra por nombre normalizado + encadenamiento de códigos con verificación de continuidad de VU** | Usar `codigo_negocio` y aceptar historias truncadas (Global X, Sumar, Fidugob, Itaú, Occitesoros pierden 8–10 años) | Otra | ☑ Opción 1 |
| **D2** | Fuente de categorías | **Clasificación derivada (vol + corr con COLCAP/Global/Fiducuenta, §2.1), revisada manualmente** | Solo regex por nombre (documentado que falla en ≥ 4 casos) | Clasificación manual de Carlos fondo a fondo | ☑ Opción 1 |
| **D3** | Universo | **18 core + 3 late (§2.3), entradas punto-en-el-tiempo** | Solo los 13 con historia desde 2016-01 (opción A) | 21 fijos desde 2021-07 (opción C) | ☑ Opción 1 |
| **D4** | Sustituciones puntuales | Ninguna | Sustituir `59079` (híbrido) por `3644` (LATAM) en RV_INTL | Sustituir `61583` por `51959` en RF_LARGO / otras (indicar) | ☑ Opción 1 |
| **D5** | Mecanismo de bandas | **HRP sin restricciones → proyección QP al poliedro de bandas (cvxpy)** | HRP jerárquico: pesos de categoría fijos por perfil + HRP intra-categoría | Bandas solo como filtro ex-post (rechazar y re-optimizar) | ☑ Opción 1 |
| **D6** | Bandas numéricas | **Conservador 20/70, Moderado 45/40, Agresivo 70/20 (RV máx / RF mín), intl 10/20/35, máx por fondo 25/20/20 (§3.2)** | Más conservadoras: 15/75, 35/50, 60/30 | Más agresivas: 25/65, 55/35, 85/10 | ☑ Opción 1 |
| **D7** | Look-through de mixtos | **50 % RV / 50 % RF fijo** | Beta rolling del mixto vs. COLCAP+Global (estimada en la ventana) | Contar mixtos 100 % como RV (conservador) | ☑ Opción 1 |
| **D8** | Ventana de estimación | **Rolling 24 m, retornos diarios hábiles, Ledoit-Wolf; sensibilidad 12/36 m y expanding** | Rolling 36 m | Expanding desde 2018-01 | ☑ Opción 1 |
| **D9** | Inicio OOS / rebalanceo | **Opción B: 1er rebalanceo 2020-01, trimestral (26 puntos); mensual como sensibilidad (80)** | Opción A: 2018-01, trimestral (34 puntos, universo 13) | Opción B'': 2019-01 con ventana 12 m (30 puntos) | ☑ Opción 1 |
| **D10** | Sesgo de supervivencia | **Documentar + test "R" con 2 fondos muertos** | Solo documentar | Universo point-in-time completo (incluir todos los fondos vivos en cada fecha; ~40–60 fondos) | ☑ Opción 1 |
| **D11** | App token Socrata | **Registrar token (gratuito) antes de Fase 1** | Seguir sin token (particionar por año/fondo; riesgo de timeouts) | — | ☑ Opción 1 |

**Bloqueantes:** ninguno. **Condición de salida del gate:** D1, D3, D6, D8 y D9 aprobadas explícitamente por Carlos. ✅ Cumplida el 2026-09-11.

---

## Anexo A — Queries usadas (evidencia)

Base: `https://www.datos.gov.co/resource/{id}.json`. Parámetros mostrados decodificados. Scripts en `docs/fase0_evidence/scripts/`.

| Ref | Dataset | Query (parámetros SoQL) | Resultado clave | Script |
|---|---|---|---|---|
| Q0 | ambos | `sodapy.Socrata.get_metadata(id)` | 26 columnas idénticas; A: 2,918,294 filas, 2016-01-01 → 2026-09-08 | `q_meta.py` |
| Q1 | ambos | `$select=min(fecha_corte) as fmin, max(fecha_corte) as fmax, count(*) as n` | A: 2016-01-01 / 2026-09-08 / 2,918,294 · B: 2024-01-01 / 2026-09-08 / 980,944 | `q2.py` |
| Q2 | ambos | `$select=subtipo_negocio, nombre_subtipo_patrimonio, count(*) as n, count(distinct codigo_negocio) as n_codigos&$group=subtipo_negocio, nombre_subtipo_patrimonio` | A: 570/16/28/7/401 códigos por subtipo (37 s) · B: 300/7/16/6/292 | `q1.py`, `q2.py` |
| Q3 | ambos | `$select=date_trunc_y(fecha_corte) as anio, count(*) as n_rows, count(distinct fecha_corte) as n_fechas, count(distinct codigo_negocio) as n_codigos&$group=anio` | Filas/año idénticas A≡B para 2024–2026; 365/366 fechas por año; 420→483 códigos/año | `q1.py`, `q2.py` |
| Q4 | A | `$select=fecha_corte, count(*) as n&$group=fecha_corte&$limit=50000` | 3,904 fechas, 0 faltantes, 104–105 fines de semana/año, filas/fecha uniformes L–D | `q3.py` |
| Q5 | A | `$where=fecha_corte='2026-09-08T00:00:00.000'&$limit=5000` | 1,037 filas, 447 códigos, 1,031 claves, 6 duplicados; AUM por clase; top-30 por AUM | `q4.py` |
| Q6 | A | `$where=codigo_negocio=2852 AND tipo_participacion=800 AND fecha_corte between '2026-08-20' and '2026-09-08'` (ídem `60678/800`, y `2025-08-01…`) | Fórmulas de `rentabilidad_diaria/mensual/semestral/anual` reproducidas exactamente | `q5.py` |
| Q7b | A | por año: `$select=codigo_negocio, nombre_patrimonio, codigo_entidad, nombre_subtipo_patrimonio, min(fecha_corte), max(fecha_corte), count(*)&$where=fecha_corte between 'YYYY-01-01' and 'YYYY-12-31'&$group=…` | 1,013 códigos, 944 nombres, 71 traspasos de código, 0 códigos multi-entidad, 448 vivos / 565 muertos | `q6b.py`, `q6c.py` |
| Q8 | A | por código (54 códigos): `$select=fecha_corte,…,valor_unidad_operaciones,…&$where=codigo_negocio={c}&$order=fecha_corte,tipo_participacion&$limit=50000` | 490 k filas; completitud, brechas, duplicados, continuidad en traspasos | `q8_history.py` |
| Q9 | — | cálculo local sobre Q8 | Tabla de completitud n=47; vol/MDD diarios; corr semanal | `q9_metrics.py` |
| Q10 | A | `$select=fecha_corte,codigo_negocio,…,valor_unidad_operaciones,valor_fondo_cierre_dia_t&$where=date_extract_d(fecha_corte)=1 AND subtipo_negocio!=7&$limit=50000&$offset=…` | 68,185 filas, 606 códigos (valores de unidad de inicio de mes, todas las clases) | `q10_monthly.py` |
| Q11 | — | cálculo local sobre Q10 | Clasificación vol/corr de 315 fondos; universo propuesto | `q11_classify.py` |
| RL | A | 40 × `$select=fecha_corte,codigo_negocio&$limit=100&$offset=i·100` secuencial + 60 × en 20 hilos | 100 % HTTP 200, 0 × 429, latencia 0.9 s media | inline (§1.0) |

## Anexo B — Archivos de evidencia

- `data/dates.csv` — 3,904 fechas con filas por fecha.
- `data/lifetimes_by_year.csv`, `data/code_lifetimes.csv` — vida de cada `codigo_negocio` (base del análisis de reasignación).
- `data/chains.json` — cadenas de códigos por fondo candidato.
- `data/metrics.csv` — completitud y métricas diarias de los 47 candidatos.
- `data/classified_all.csv` — 315 fondos con vol/corr/MDD/AUM y clase derivada.
- `data/universe_proposed.csv` — universo propuesto (core/late/alt).
- `data/candidates_snapshot.csv` — 223 FIC vivos a 2026-09-08 con AUM agregado y clase por nombre.
- No se incluyen las descargas crudas (`hist/`, 93 MB; `monthly_all.csv`, 11 MB): se regeneran con `q8_history.py` y `q10_monthly.py` en ~5 min.

## Next Steps

1. ~~Carlos revisa §5 y marca D1–D11~~ — hecho 2026-09-11.
2. Si se aprueba D11: registrar app token en datos.gov.co y guardarlo en `.env` (no en el repo) — Est. 0.5 h.
3. ~~Fase 1 = tabla maestra de fondos (D1) + loader con dedupe y calendario hábil (§1.3)~~ — hecho 2026-09-11: `data/ingest_sfc.py`, 21/21 fondos pasan el gate de suficiencia, ver `manifest.json`.
