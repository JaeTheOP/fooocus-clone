<?php

declare(strict_types=1);

return static function ($db, array $context = []): array {
    $root = rtrim((string)($context['project_root'] ?? dirname(__DIR__, 2)), DIRECTORY_SEPARATOR);
    $adminPath = $root . '/admin/pricing_manager.php';
    $headerPath = $root . '/admin/includes/header.php';
    $catalogPath = $root . '/includes/data/ajax_tier2_catalog.php';
    $backupDir = $root . '/storage/pricing-backups';
    $versionPath = $root . '/includes/app_version.php';

    $adminText = is_file($adminPath) ? (string)file_get_contents($adminPath) : '';
    $headerText = is_file($headerPath) ? (string)file_get_contents($headerPath) : '';
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

    $versionText = is_file($versionPath) ? (string)file_get_contents($versionPath) : '';
    $discountMarker = $backupDir . '/equipment-price-reduction-2.0.0.json';
    $managerMarker = $backupDir . '/pricing-manager-2.1.0.json';

    return [
        [
            'name' => 'Pricing Manager admin page',
            'ok' => is_file($adminPath) && strpos($adminText, 'KLEVR Pricing Manager') !== false,
            'critical' => true,
            'message' => is_file($adminPath) ? 'The Pricing Manager admin page is installed.' : 'The Pricing Manager admin page is missing.',
        ],
        [
            'name' => 'Pricing Manager navigation',
            'ok' => strpos($headerText, 'pricing_manager.php') !== false,
            'critical' => true,
            'message' => strpos($headerText, 'pricing_manager.php') !== false ? 'Admin navigation links to Pricing Manager.' : 'Admin navigation does not link to Pricing Manager.',
        ],
        [
            'name' => 'shared AJAX pricing catalog',
            'ok' => $catalogOk && $catalogCount >= 100,
            'critical' => true,
            'message' => $catalogOk ? 'The shared catalog exposes ' . $catalogCount . ' customer prices.' : 'The shared AJAX pricing catalog could not be loaded.',
        ],
        [
            'name' => 'equipment reduction baseline',
            'ok' => is_file($discountMarker),
            'critical' => true,
            'message' => is_file($discountMarker) ? 'The cumulative 2.0.0 equipment-price baseline is present.' : 'The 2.0.0 equipment-price marker is missing.',
        ],
        [
            'name' => 'pricing backup storage',
            'ok' => is_dir($backupDir) && is_writable($backupDir) && is_file($managerMarker),
            'critical' => true,
            'message' => is_dir($backupDir) && is_writable($backupDir) ? 'Protected pricing backup storage is writable.' : 'Pricing backup storage is missing or not writable.',
        ],
        [
            'name' => 'application version',
            'ok' => strpos($versionText, '2.1.1') !== false,
            'critical' => true,
            'message' => strpos($versionText, '2.1.1') !== false ? 'KLEVR application version is 2.1.1.' : 'The application version is not 2.1.1.',
        ],
        [
            'name' => 'database connection',
            'ok' => $db instanceof PDO,
            'critical' => true,
            'message' => $db instanceof PDO ? 'The updater supplied the expected PDO connection.' : 'The updater did not supply a PDO connection.',
        ],
    ];
};
