<?php

declare(strict_types=1);

return static function (PDO $db, array $context = []): array {
    $root = rtrim((string)($context['project_root'] ?? dirname(__DIR__, 2)), DIRECTORY_SEPARATOR);
    $backupDir = $root . '/storage/pricing-backups';
    if (!is_dir($backupDir) && !mkdir($backupDir, 0775, true) && !is_dir($backupDir)) {
        throw new RuntimeException('Pricing backup directory could not be created.');
    }

    $headerPath = $root . '/admin/includes/header.php';
    $navAdded = false;
    $headerBackup = null;
    $begin = '<!-- KLEVR_PRICING_MANAGER_NAV_BEGIN -->';
    $end = '<!-- KLEVR_PRICING_MANAGER_NAV_END -->';
    $link = $begin . "\n"
        . '<a href="pricing_manager.php" class="nav-link"><span aria-hidden="true">$</span><span>Pricing Manager</span></a>' . "\n"
        . $end;

    if (is_file($headerPath) && is_writable($headerPath)) {
        $header = (string)file_get_contents($headerPath);
        if (strpos($header, $begin) === false && strpos($header, 'pricing_manager.php') === false) {
            $headerBackup = $backupDir . '/admin-header.pre-pricing-manager.' . gmdate('YmdHis') . '.php';
            if (!copy($headerPath, $headerBackup)) {
                throw new RuntimeException('Admin navigation backup could not be created.');
            }

            $patterns = [
                '/(<a\b[^>]*href=["\'][^"\']*ajax_pricing\.php[^"\']*["\'][\s\S]*?<\/a>)/i',
                '/(<a\b[^>]*href=["\'][^"\']*products\.php[^"\']*["\'][\s\S]*?<\/a>)/i',
            ];
            $updated = $header;
            foreach ($patterns as $pattern) {
                if (preg_match($pattern, $updated)) {
                    $updated = (string)preg_replace($pattern, '$1' . "\n" . $link, $updated, 1);
                    break;
                }
            }
            if ($updated === $header && stripos($header, '</nav>') !== false) {
                $updated = (string)preg_replace('/<\/nav>/i', $link . "\n</nav>", $header, 1);
            }
            if ($updated === $header) {
                $updated .= "\n" . $link . "\n";
            }

            $temp = $headerPath . '.pricing-manager.tmp';
            if (file_put_contents($temp, $updated, LOCK_EX) === false || !rename($temp, $headerPath)) {
                @unlink($temp);
                throw new RuntimeException('Pricing Manager navigation link could not be installed.');
            }
            $navAdded = true;
        }
    }

    $marker = $backupDir . '/pricing-manager-2.1.0.json';
    file_put_contents($marker, json_encode([
        'installed_at_utc' => gmdate('c'),
        'navigation_added' => $navAdded,
        'header_backup' => $headerBackup,
        'database_schema_changed' => false,
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES), LOCK_EX);

    return [
        'pricing_backup_directory' => $backupDir,
        'navigation_added' => $navAdded,
        'database_schema_changed' => false,
    ];
};
