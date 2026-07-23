<?php

declare(strict_types=1);

return static function (PDO $db, array $context = []): array {
    $root = rtrim((string)($context['project_root'] ?? dirname(__DIR__, 2)), DIRECTORY_SEPARATOR);
    $catalogPath = $root . '/includes/data/ajax_tier2_catalog.php';
    if (!is_file($catalogPath)) {
        throw new RuntimeException('AJAX pricing catalog not found.');
    }

    $catalog = require $catalogPath;
    if (!is_array($catalog)) {
        throw new RuntimeException('AJAX pricing catalog is invalid.');
    }

    $backupDir = $root . '/storage/pricing-backups';
    if (!is_dir($backupDir)) {
        mkdir($backupDir, 0775, true);
    }
    $marker = $backupDir . '/equipment-price-reduction-2.0.0.json';
    if (is_file($marker)) {
        return ['already_applied' => true];
    }

    $backup = $backupDir . '/ajax_tier2_catalog.pre-2.0.0.' . gmdate('YmdHis') . '.php';
    if (!copy($catalogPath, $backup)) {
        throw new RuntimeException('AJAX pricing backup could not be created.');
    }

    $count = 0;
    $apply = static function (&$node) use (&$apply, &$count): void {
        if (!is_array($node)) {
            return;
        }
        if (isset($node['customer_price']) && is_numeric($node['customer_price'])) {
            $node['customer_price'] = round((float)$node['customer_price'] * 0.90, 2);
            $count++;
        }
        foreach ($node as &$value) {
            if (is_array($value)) {
                $apply($value);
            }
        }
        unset($value);
    };
    $apply($catalog);

    if ($count < 100) {
        throw new RuntimeException('Expected AJAX customer prices were not found.');
    }

    $php = "<?php\n\ndeclare(strict_types=1);\n\nreturn " . var_export($catalog, true) . ";\n";
    $temp = $catalogPath . '.tmp';
    if (file_put_contents($temp, $php, LOCK_EX) === false || !rename($temp, $catalogPath)) {
        @unlink($temp);
        throw new RuntimeException('Discounted AJAX pricing catalog could not be activated.');
    }

    $addons = 0;
    try {
        $rows = $db->query('SELECT id, price, sale_price FROM hardware_addons')->fetchAll(PDO::FETCH_ASSOC);
        $update = $db->prepare('UPDATE hardware_addons SET price = ?, sale_price = ? WHERE id = ?');
        foreach ($rows as $row) {
            if (!is_numeric($row['price'] ?? null)) {
                continue;
            }
            $sale = is_numeric($row['sale_price'] ?? null) ? round((float)$row['sale_price'] * 0.90, 2) : null;
            $update->execute([round((float)$row['price'] * 0.90, 2), $sale, $row['id']]);
            $addons++;
        }
    } catch (Throwable $ignored) {
        $addons = 0;
    }

    file_put_contents($marker, json_encode([
        'discount_percent' => 10,
        'catalog_items' => $count,
        'hardware_addons' => $addons,
        'catalog_backup' => $backup,
        'applied_at_utc' => gmdate('c'),
    ], JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES), LOCK_EX);

    return ['catalog_items_repriced' => $count, 'hardware_addons_repriced' => $addons];
};
