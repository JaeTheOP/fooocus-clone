<?php

declare(strict_types=1);

return static function ($db, array $context = []): array {
    $root = rtrim((string)($context['project_root'] ?? dirname(__DIR__, 2)), DIRECTORY_SEPARATOR);
    $catalogPath = $root . '/includes/data/ajax_tier2_catalog.php';
    $versionPath = $root . '/includes/app_version.php';
    $markerPath = $root . '/storage/pricing-backups/equipment-price-reduction-2.0.0.json';

    $catalogCount = 0;
    $catalogOk = false;
    if (is_file($catalogPath)) {
        $catalog = require $catalogPath;
        $catalogOk = is_array($catalog);
        if ($catalogOk) {
            $count = static function ($node) use (&$count, &$catalogCount): void {
                if (!is_array($node)) {
                    return;
                }
                if (isset($node['customer_price']) && is_numeric($node['customer_price'])) {
                    $catalogCount++;
                }
                foreach ($node as $value) {
                    if (is_array($value)) {
                        $count($value);
                    }
                }
            };
            $count($catalog);
        }
    }

    $markerOk = false;
    if (is_file($markerPath)) {
        $marker = json_decode((string)file_get_contents($markerPath), true);
        $markerOk = is_array($marker)
            && (int)($marker['discount_percent'] ?? 0) === 10
            && (int)($marker['catalog_items'] ?? 0) >= 100;
    }

    $versionText = is_file($versionPath) ? (string)file_get_contents($versionPath) : '';

    return [
        [
            'name' => 'discounted AJAX equipment catalog',
            'ok' => $catalogOk && $catalogCount >= 100,
            'critical' => true,
            'message' => $catalogOk
                ? 'The AJAX catalog contains ' . $catalogCount . ' customer equipment prices.'
                : 'The AJAX catalog could not be loaded.',
        ],
        [
            'name' => 'equipment reduction marker',
            'ok' => $markerOk,
            'critical' => true,
            'message' => $markerOk
                ? 'The 10 percent equipment reduction completed.'
                : 'The equipment reduction marker is missing or incomplete.',
        ],
        [
            'name' => 'application version',
            'ok' => strpos($versionText, '2.0.0') !== false,
            'critical' => true,
            'message' => strpos($versionText, '2.0.0') !== false
                ? 'KLEVR application version is 2.0.0.'
                : 'The application version file does not contain 2.0.0.',
        ],
        [
            'name' => 'database connection',
            'ok' => $db instanceof PDO,
            'critical' => true,
            'message' => $db instanceof PDO
                ? 'The updater supplied the expected PDO connection.'
                : 'The updater did not supply a PDO connection.',
        ],
    ];
};
