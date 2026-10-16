DROP MATERIALIZED VIEW IF EXISTS huella_digital.hdm_resultados_base_laicpms;

CREATE MATERIALIZED VIEW huella_digital.hdm_resultados_base_laicpms AS

WITH laicpms AS (

    SELECT
        codigo_muestra,
        elemento,

        /* =====================================================
           NORMALIZACIÓN DE UNIDAD
           CPS  → cps
           RESTO → ppm
           ===================================================== */
        CASE
            WHEN LOWER(TRIM(unidad)) = 'cps'
                THEN 'cps'
            ELSE 'ppm'
        END AS unidad,

        /* =====================================================
           MEDIANA DE LA COLUMNA valor
           DE hdm_resultados_base
           ===================================================== */
        PERCENTILE_CONT(0.5)
        WITHIN GROUP (
            ORDER BY valor::DOUBLE PRECISION
        )::NUMERIC AS valor,

        /* =====================================================
           RESTO DE CAMPOS
           ===================================================== */
        MIN(tipo_muestra) AS tipo_muestra,
        MIN(subtipo_muestra) AS subtipo_muestra,
        MIN(fecha_muestreo) AS fecha_muestreo,
        MIN(wgs84_latitud) AS wgs84_latitud,
        MIN(wgs84_longitud) AS wgs84_longitud,
        MIN(entidad) AS entidad,
        MIN(direccion_tecnica) AS direccion_tecnica,
        MIN(proyecto) AS proyecto,
        MIN(laboratorio_nombre) AS laboratorio_nombre,
        MIN(tipo_muestra_lab) AS tipo_muestra_lab,
        MIN(codigo_muestra_lab) AS codigo_muestra_lab,
        MIN(fecha_envio_solicitud) AS fecha_envio_solicitud,
        MIN(orden_servicio) AS orden_servicio,
        MIN(fecha_orden_servicio) AS fecha_orden_servicio,
        MIN(batchlab) AS batchlab,
        MIN(numero_reporte_glq) AS numero_reporte_glq,
        MIN(fecha_reporte) AS fecha_reporte,
        MIN(codigo_tecnica) AS codigo_tecnica,
        MIN(tecnica_analitica) AS tecnica_analitica,
        MIN(codigo_preparacion) AS codigo_preparacion,
        MIN(secado) AS secado,
        MIN(tamizado) AS tamizado,
        MIN(trituracion) AS trituracion,
        MIN(pulverizacion) AS pulverizacion,
        MIN(malla_final) AS malla_final,
        MIN(pretratamiento) AS pretratamiento,
        MIN(ataque) AS ataque,
        MIN(alcance_ataque) AS alcance_ataque,
        MIN(descripcion) AS descripcion,
        MIN(tipo_valor) AS tipo_valor,
        MIN(municipio) AS municipio,
        MIN(departamento) AS departamento,
        MIN(distrito) AS distrito

    FROM huella_digital.hdm_resultados_base

    WHERE LOWER(
        TRIM(
            REPLACE(
                tecnica_analitica,
                CHR(160),
                ' '
            )
        )
    ) =
    LOWER(
        'Espectrometría de masas con plasma acoplado inductivamente mediante ablasión láser: [LA-ICP-MS]'
    )

    /* =========================================================
       AGRUPAR POR:
       Muestra + Elemento + Unidad normalizada
       ========================================================= */
    GROUP BY
        codigo_muestra,
        elemento,
        CASE
            WHEN LOWER(TRIM(unidad)) = 'cps'
                THEN 'cps'
            ELSE 'ppm'
        END
)

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
    unidad,
    valor,
    tipo_valor,
    elemento,
    municipio,
    departamento,
    distrito

FROM laicpms;