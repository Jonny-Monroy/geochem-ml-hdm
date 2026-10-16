DROP MATERIALIZED VIEW IF EXISTS huella_digital.hdm_resultados_base;

CREATE MATERIALIZED VIEW huella_digital.hdm_resultados_base AS

WITH datos_base AS (

    SELECT
        a.codigo_muestra,
        a.tipo_muestra,
        a.subtipo_muestra,
        a.fecha_muestreo,
        a.wgs84_latitud,
        a.wgs84_longitud,
        a.entidad,
        a.direccion_tecnica,
        a.proyecto,
        a.laboratorio_nombre,
        a.tipo_muestra_lab,
        a.codigo_muestra_lab,
        a.fecha_envio_solicitud,
        a.orden_servicio,
        a.fecha_orden_servicio,
        a.batchlab,
        a.numero_reporte_glq,
        a.fecha_reporte,
        a.codigo_tecnica,
        a.tecnica_analitica,
        a.codigo_preparacion,
        a.secado,
        a.tamizado,
        a.trituracion,
        a.pulverizacion,
        a.malla_final,
        a.pretratamiento,
        a.ataque,
        a.alcance_ataque,
        a.descripcion,
        a.valor_original,
        a.unidad,

        /* =====================================================
           VALOR NUMÉRICO EXTRAÍDO DE valor_original
           
           Se utiliza para:
           - CPS
           - cálculo de valor_ppm
           
           Ejemplos:
           < 0,0004 → 0.0004
           > 500    → 500
           0,25     → 0.25
           ===================================================== */
        CASE

            WHEN REPLACE(TRIM(a.valor_original), ' ', '')
                 ~ '^<[0-9]+([,.][0-9]+)?$'
            THEN
                REPLACE(
                    SUBSTRING(
                        REPLACE(
                            TRIM(a.valor_original),
                            ' ',
                            ''
                        )
                        FROM 2
                    ),
                    ',',
                    '.'
                )::NUMERIC

            WHEN REPLACE(TRIM(a.valor_original), ' ', '')
                 ~ '^>[0-9]+([,.][0-9]+)?$'
            THEN
                REPLACE(
                    SUBSTRING(
                        REPLACE(
                            TRIM(a.valor_original),
                            ' ',
                            ''
                        )
                        FROM 2
                    ),
                    ',',
                    '.'
                )::NUMERIC

            WHEN REPLACE(TRIM(a.valor_original), ' ', '')
                 ~ '^[0-9]+([,.][0-9]+)?$'
            THEN
                REPLACE(
                    REPLACE(
                        TRIM(a.valor_original),
                        ' ',
                        ''
                    ),
                    ',',
                    '.'
                )::NUMERIC

            ELSE NULL

        END AS valor_numerico_original,

        /* =====================================================
           FACTOR DE CONVERSIÓN DESDE dominio_valor
           ===================================================== */
        f.factor,

        a.tipo_valor,
        a.elemento,

        /* =====================================================
           INFORMACIÓN ESPACIAL
           ===================================================== */
        d.municipio,
        d.departamento,
        d.distrito

    FROM huella_digital.hdm_atlas a

    /* =========================================================
       OBTENER FACTOR DESDE dominio_valor
       
       Ejemplos:
       ppm      [1]
       ppb      [0,001]
       per      [10000]
       mg_Kg    [1]
       ========================================================= */
    LEFT JOIN LATERAL (

        SELECT
            REPLACE(
                (
                    REGEXP_MATCH(
                        dv.valor,
                        '\[([0-9]+([,.][0-9]+)?)\]'
                    )
                )[1],
                ',',
                '.'
            )::NUMERIC AS factor

        FROM huella_digital.dominio_valor dv

        WHERE LOWER(
            TRIM(
                SPLIT_PART(
                    dv.valor,
                    ':',
                    1
                )
            )
        ) =
        LOWER(
            TRIM(a.unidad)
        )

        ORDER BY
            dv.creado DESC NULLS LAST,
            dv.id_dominio_valor DESC

        LIMIT 1

    ) f ON TRUE

    /* =========================================================
       JOIN ESPACIAL
       Máximo un distrito por registro de hdm_atlas
       ========================================================= */
    LEFT JOIN LATERAL (

        SELECT
            d.municipio,
            d.departamento,
            d.distrito

        FROM huella_digital.distritos d

        WHERE ST_Intersects(

            ST_SetSRID(

                ST_MakePoint(

                    NULLIF(
                        REPLACE(
                            a.wgs84_longitud,
                            ',',
                            '.'
                        ),
                        ''
                    )::DOUBLE PRECISION,

                    NULLIF(
                        REPLACE(
                            a.wgs84_latitud,
                            ',',
                            '.'
                        ),
                        ''
                    )::DOUBLE PRECISION

                ),
                4326

            ),

            d.geom

        )

        ORDER BY ST_Area(d.geom)

        LIMIT 1

    ) d ON TRUE
),

/* =============================================================
   1. CALCULAR valor_ppm
   ============================================================= */

datos_ppm AS (

    SELECT
        db.*,

        CASE

            /* =================================================
               CPS
               No se convierte a PPM
               ================================================= */
            WHEN LOWER(TRIM(db.unidad)) = 'cps'
            THEN NULL

            /* =================================================
               <
               √valor × factor
               ================================================= */
            WHEN REPLACE(
                    TRIM(db.valor_original),
                    ' ',
                 ''
                 ) ~ '^<[0-9]+([,.][0-9]+)?$'
                 AND db.factor IS NOT NULL

            THEN
                SQRT(
                    db.valor_numerico_original
                ) * db.factor

            /* =================================================
               >
               NO raíz cuadrada
               valor × factor
               ================================================= */
            WHEN REPLACE(
                    TRIM(db.valor_original),
                    ' ',
                 ''
                 ) ~ '^>[0-9]+([,.][0-9]+)?$'
                 AND db.factor IS NOT NULL

            THEN
                db.valor_numerico_original * db.factor

            /* =================================================
               Valor numérico normal
               valor × factor
               ================================================= */
            WHEN db.valor_numerico_original IS NOT NULL
                 AND db.factor IS NOT NULL

            THEN
                db.valor_numerico_original * db.factor

            /* =================================================
               SD, LD, textos, etc.
               ================================================= */
            ELSE NULL

        END AS valor_ppm

    FROM datos_base db
),

/* =============================================================
   2. CALCULAR LOGARITMO PARA CLR
   Solo unidades diferentes de CPS y valor_ppm > 0
   ============================================================= */

datos_log AS (

    SELECT
        dp.*,

        CASE

            WHEN LOWER(TRIM(dp.unidad)) <> 'cps'
                 AND dp.valor_ppm > 0

            THEN LN(dp.valor_ppm)

            ELSE NULL

        END AS log_x

    FROM datos_ppm dp
),

/* =============================================================
   3. CALCULAR PROMEDIO DE LOGARITMOS
   AGRUPADO POR codigo_muestra_lab
   ============================================================= */

datos_promedio AS (

    SELECT
        dl.*,

        AVG(dl.log_x) OVER (
            PARTITION BY dl.codigo_muestra_lab
        ) AS promedio_log

    FROM datos_log dl
)

/* =============================================================
   4. RESULTADO FINAL
   ============================================================= */

SELECT
    codigo_muestra,
    tipo_muestra,
    subtipo_muestra,
    fecha_muestreo,
    wgs84_latitud,
    wgs84_longitud,
    entidad,
    direccion_tecnica,
    proyecto,
    laboratorio_nombre,
    tipo_muestra_lab,
    codigo_muestra_lab,
    fecha_envio_solicitud,
    orden_servicio,
    fecha_orden_servicio,
    batchlab,
    numero_reporte_glq,
    fecha_reporte,
    codigo_tecnica,
    tecnica_analitica,
    codigo_preparacion,
    secado,
    tamizado,
    trituracion,
    pulverizacion,
    malla_final,
    pretratamiento,
    ataque,
    alcance_ataque,
    descripcion,
    valor_original,
    /* =========================================================
       NORMALIZACIÓN DE UNIDAD PARA LA INTERFAZ DE SHINY
       ========================================================= */
    CASE
        WHEN LOWER(TRIM(unidad)) = 'cps' THEN 'cps'
        ELSE 'ppm'
    END AS unidad,

    /* =========================================================
       valor_ppm
       ========================================================= */
    valor_ppm,

    /* =========================================================
       valor
       
       CPS:
           LN(valor_original)

       NO CPS:
           CLR = LN(valor_ppm) - promedio_log
       ========================================================= */
    CASE

        /* =====================================================
           CPS
           ===================================================== */
        WHEN LOWER(TRIM(unidad)) = 'cps'
             AND valor_numerico_original > 0

        THEN
            LN(valor_numerico_original)

        /* =====================================================
           NO CPS
           CLR
           ===================================================== */
        WHEN LOWER(TRIM(unidad)) <> 'cps'
             AND valor_ppm > 0
             AND promedio_log IS NOT NULL

        THEN
            log_x - promedio_log

        ELSE NULL

    END AS valor,

    tipo_valor,
    elemento,
    municipio,
    departamento,
    distrito

FROM datos_promedio;