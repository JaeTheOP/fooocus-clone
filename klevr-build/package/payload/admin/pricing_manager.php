<?php

declare(strict_types=1);

$page_title = 'Pricing Manager';
$root = dirname(__DIR__);

foreach ([$root . '/includes/config.php', $root . '/config.php'] as $configFile) {
    if (is_file($configFile)) {
        require_once $configFile;
        break;
    }
}

if (session_status() !== PHP_SESSION_ACTIVE) {
    @session_start();
}

$headerHtml = '';
$headerPath = __DIR__ . '/includes/header.php';
if (is_file($headerPath)) {
    ob_start();
    require $headerPath;
    $headerHtml = (string)ob_get_clean();
}

function klevr_pricing_h($value): string
{
    return htmlspecialchars((string)$value, ENT_QUOTES, 'UTF-8');
}

function klevr_pricing_db(): ?PDO
{
    foreach (['pdo', 'db', 'database', 'conn', 'connection'] as $key) {
        if (($GLOBALS[$key] ?? null) instanceof PDO) {
            return $GLOBALS[$key];
        }
    }
    if (function_exists('db')) {
        try {
            $db = db();
            if ($db instanceof PDO) {
                return $db;
            }
        } catch (Throwable $ignored) {
        }
    }
    return null;
}

function klevr_pricing_csrf_token(): string
{
    if (function_exists('csrf_token')) {
        return (string)csrf_token();
    }
    if (empty($_SESSION['klevr_pricing_csrf'])) {
        $_SESSION['klevr_pricing_csrf'] = bin2hex(random_bytes(32));
    }
    return (string)$_SESSION['klevr_pricing_csrf'];
}

function klevr_pricing_verify_csrf(string $token): bool
{
    if (function_exists('verify_csrf_token')) {
        try {
            return (bool)verify_csrf_token($token);
        } catch (Throwable $ignored) {
            return false;
        }
    }
    return isset($_SESSION['klevr_pricing_csrf'])
        && hash_equals((string)$_SESSION['klevr_pricing_csrf'], $token);
}

function klevr_pricing_catalog_path(string $root): string
{
    return rtrim($root, DIRECTORY_SEPARATOR) . '/includes/data/ajax_tier2_catalog.php';
}

function klevr_pricing_load_catalog(string $path): array
{
    if (!is_file($path)) {
        throw new RuntimeException('The AJAX pricing catalog was not found.');
    }
    $catalog = require $path;
    if (!is_array($catalog)) {
        throw new RuntimeException('The AJAX pricing catalog is invalid.');
    }
    return $catalog;
}

function klevr_pricing_entry_name(array $node, array $path): string
{
    foreach (['name', 'title', 'product_name', 'label', 'description', 'sku', 'model'] as $field) {
        if (isset($node[$field]) && is_scalar($node[$field]) && trim((string)$node[$field]) !== '') {
            return trim((string)$node[$field]);
        }
    }
    $last = end($path);
    return ucwords(str_replace(['_', '-'], ' ', (string)$last));
}

function klevr_pricing_collect_catalog(array $node, array $path = [], array &$entries = []): array
{
    if (isset($node['customer_price']) && is_numeric($node['customer_price'])) {
        $entries[] = [
            'path' => $path,
            'name' => klevr_pricing_entry_name($node, $path),
            'sku' => isset($node['sku']) && is_scalar($node['sku']) ? trim((string)$node['sku']) : '',
            'cost' => is_numeric($node['cost'] ?? null) ? (float)$node['cost'] : (is_numeric($node['price'] ?? null) ? (float)$node['price'] : null),
            'customer_price' => (float)$node['customer_price'],
            'installation_price' => is_numeric($node['installation_price'] ?? null) ? (float)$node['installation_price'] : null,
            'node' => $node,
        ];
    }
    foreach ($node as $key => $value) {
        if (is_array($value)) {
            klevr_pricing_collect_catalog($value, array_merge($path, [$key]), $entries);
        }
    }
    return $entries;
}

function klevr_pricing_token(array $path): string
{
    $json = json_encode($path, JSON_UNESCAPED_SLASHES);
    return rtrim(strtr(base64_encode((string)$json), '+/', '-_'), '=');
}

function klevr_pricing_path_from_token(string $token): ?array
{
    $padding = strlen($token) % 4;
    if ($padding) {
        $token .= str_repeat('=', 4 - $padding);
    }
    $decoded = base64_decode(strtr($token, '-_', '+/'), true);
    if ($decoded === false) {
        return null;
    }
    $path = json_decode($decoded, true);
    return is_array($path) ? $path : null;
}

function klevr_pricing_set_catalog_price(array &$catalog, array $path, float $price): bool
{
    $ref =& $catalog;
    foreach ($path as $key) {
        if (!is_array($ref) || !array_key_exists($key, $ref)) {
            return false;
        }
        $ref =& $ref[$key];
    }
    if (!is_array($ref) || !array_key_exists('customer_price', $ref)) {
        return false;
    }
    $ref['customer_price'] = round($price, 2);
    return true;
}

function klevr_pricing_write_catalog(string $catalogPath, array $catalog, string $backupDir): string
{
    if (!is_dir($backupDir) && !mkdir($backupDir, 0775, true) && !is_dir($backupDir)) {
        throw new RuntimeException('The pricing backup directory could not be created.');
    }
    if (!is_writable(dirname($catalogPath))) {
        throw new RuntimeException('The catalog directory is not writable.');
    }

    $backup = $backupDir . '/ajax_tier2_catalog.' . gmdate('Ymd-His') . '.' . bin2hex(random_bytes(3)) . '.php';
    if (!copy($catalogPath, $backup)) {
        throw new RuntimeException('The current pricing catalog could not be backed up.');
    }

    $php = "<?php\n\ndeclare(strict_types=1);\n\nreturn " . var_export($catalog, true) . ";\n";
    $temp = $catalogPath . '.pricing-manager.' . bin2hex(random_bytes(4)) . '.tmp';
    if (file_put_contents($temp, $php, LOCK_EX) === false) {
        throw new RuntimeException('The new pricing catalog could not be written.');
    }
    if (!@rename($temp, $catalogPath)) {
        @unlink($temp);
        throw new RuntimeException('The new pricing catalog could not be activated.');
    }

    $backups = glob($backupDir . '/ajax_tier2_catalog.*.php') ?: [];
    usort($backups, static function (string $a, string $b): int {
        return (int)filemtime($b) <=> (int)filemtime($a);
    });
    foreach (array_slice($backups, 25) as $oldBackup) {
        @unlink($oldBackup);
    }

    return $backup;
}

function klevr_pricing_table_columns(PDO $db, string $table): array
{
    $allowedTables = ['products', 'packages', 'monitoring_plans', 'hardware_addons'];
    if (!in_array($table, $allowedTables, true)) {
        return [];
    }
    try {
        $rows = $db->query('SHOW COLUMNS FROM `' . $table . '`')->fetchAll(PDO::FETCH_ASSOC);
        $columns = [];
        foreach ($rows as $row) {
            if (isset($row['Field'])) {
                $columns[] = (string)$row['Field'];
            }
        }
        return $columns;
    } catch (Throwable $mysqlError) {
        try {
            $rows = $db->query("PRAGMA table_info('" . $table . "')")->fetchAll(PDO::FETCH_ASSOC);
            $columns = [];
            foreach ($rows as $row) {
                if (isset($row['name'])) {
                    $columns[] = (string)$row['name'];
                }
            }
            return $columns;
        } catch (Throwable $ignored) {
            return [];
        }
    }
}

function klevr_pricing_price_columns(array $columns): array
{
    $recognized = [
        'price', 'sale_price', 'customer_price', 'equipment_price', 'monthly_price',
        'monthly_rate', 'monitoring_price', 'installation_price', 'professional_install_price',
        'diy_price', 'activation_fee', 'service_fee', 'video_fee', 'setup_fee',
    ];
    return array_values(array_intersect($recognized, $columns));
}

function klevr_pricing_load_table(PDO $db, string $table): array
{
    $columns = klevr_pricing_table_columns($db, $table);
    if (!$columns || !in_array('id', $columns, true)) {
        return ['columns' => [], 'price_columns' => [], 'rows' => []];
    }
    $priceColumns = klevr_pricing_price_columns($columns);
    if (!$priceColumns) {
        return ['columns' => $columns, 'price_columns' => [], 'rows' => []];
    }
    $displayColumns = array_values(array_intersect(['id', 'name', 'title', 'product_name', 'sku', 'slug', 'type', 'category', 'active', 'is_active'], $columns));
    $selectColumns = array_values(array_unique(array_merge($displayColumns, $priceColumns)));
    $quoted = array_map(static function (string $column): string {
        return '`' . str_replace('`', '', $column) . '`';
    }, $selectColumns);
    try {
        $rows = $db->query('SELECT ' . implode(', ', $quoted) . ' FROM `' . $table . '` ORDER BY `id` ASC LIMIT 1000')->fetchAll(PDO::FETCH_ASSOC);
    } catch (Throwable $ignored) {
        $rows = [];
    }
    return ['columns' => $columns, 'price_columns' => $priceColumns, 'rows' => $rows];
}

function klevr_pricing_row_name(array $row, string $table): string
{
    foreach (['name', 'title', 'product_name', 'sku', 'slug'] as $field) {
        if (isset($row[$field]) && trim((string)$row[$field]) !== '') {
            return trim((string)$row[$field]);
        }
    }
    return ucwords(str_replace('_', ' ', $table)) . ' #' . (string)($row['id'] ?? '');
}

function klevr_pricing_is_package_row(string $table, array $row): bool
{
    if (in_array($table, ['packages', 'monitoring_plans'], true)) {
        return true;
    }
    $haystack = strtolower(implode(' ', array_map('strval', array_intersect_key($row, array_flip(['name', 'title', 'product_name', 'slug', 'type', 'category'])))));
    foreach (['package', 'system', 'kit', 'essential', 'signature', 'apartment', 'renter', 'rv'] as $word) {
        if (strpos($haystack, $word) !== false) {
            return true;
        }
    }
    return false;
}

function klevr_pricing_update_database(PDO $db, array $posted, array $tableData): array
{
    $changes = [];
    foreach ($posted as $table => $rows) {
        if (!isset($tableData[$table]) || !is_array($rows)) {
            continue;
        }
        $allowedColumns = $tableData[$table]['price_columns'];
        $existingRows = [];
        foreach ($tableData[$table]['rows'] as $existing) {
            $existingRows[(string)$existing['id']] = $existing;
        }
        foreach ($rows as $id => $columns) {
            $id = (string)$id;
            if (!isset($existingRows[$id]) || !is_array($columns)) {
                continue;
            }
            foreach ($columns as $column => $rawValue) {
                if (!in_array($column, $allowedColumns, true)) {
                    continue;
                }
                $rawValue = trim((string)$rawValue);
                $nullable = $column === 'sale_price';
                if ($rawValue === '' && $nullable) {
                    $newValue = null;
                } elseif (!is_numeric($rawValue)) {
                    throw new InvalidArgumentException('Every price must be a valid number.');
                } else {
                    $newValue = round((float)$rawValue, 2);
                    if ($newValue < 0 || $newValue > 1000000) {
                        throw new InvalidArgumentException('Prices must be between $0.00 and $1,000,000.00.');
                    }
                }
                $oldRaw = $existingRows[$id][$column] ?? null;
                $oldValue = is_numeric($oldRaw) ? round((float)$oldRaw, 2) : null;
                if ($oldValue === $newValue) {
                    continue;
                }
                $statement = $db->prepare('UPDATE `' . $table . '` SET `' . $column . '` = ? WHERE `id` = ?');
                $statement->execute([$newValue, $id]);
                $changes[] = [
                    'source' => 'database',
                    'table' => $table,
                    'id' => $id,
                    'name' => klevr_pricing_row_name($existingRows[$id], $table),
                    'field' => $column,
                    'old' => $oldValue,
                    'new' => $newValue,
                ];
            }
        }
    }
    return $changes;
}

function klevr_pricing_sync_catalog_item(PDO $db, array $entry, float $newPrice): array
{
    $synced = [];
    foreach (['hardware_addons', 'products', 'packages'] as $table) {
        $columns = klevr_pricing_table_columns($db, $table);
        if (!$columns || !in_array('id', $columns, true)) {
            continue;
        }
        $priceColumn = null;
        foreach (['customer_price', 'equipment_price', 'price'] as $candidate) {
            if (in_array($candidate, $columns, true)) {
                $priceColumn = $candidate;
                break;
            }
        }
        if ($priceColumn === null) {
            continue;
        }
        $matchColumn = null;
        $matchValue = null;
        if ($entry['sku'] !== '' && in_array('sku', $columns, true)) {
            $matchColumn = 'sku';
            $matchValue = $entry['sku'];
        } else {
            foreach (['name', 'title', 'product_name'] as $candidate) {
                if (in_array($candidate, $columns, true)) {
                    $matchColumn = $candidate;
                    $matchValue = $entry['name'];
                    break;
                }
            }
        }
        if ($matchColumn === null || $matchValue === null || trim((string)$matchValue) === '') {
            continue;
        }
        try {
            $statement = $db->prepare('UPDATE `' . $table . '` SET `' . $priceColumn . '` = ? WHERE LOWER(TRIM(`' . $matchColumn . '`)) = LOWER(TRIM(?))');
            $statement->execute([$newPrice, $matchValue]);
            if ($statement->rowCount() > 0) {
                $synced[] = $table . '.' . $priceColumn;
            }
        } catch (Throwable $ignored) {
        }
    }
    return $synced;
}

function klevr_pricing_append_audit(string $backupDir, array $record): void
{
    if (!is_dir($backupDir)) {
        @mkdir($backupDir, 0775, true);
    }
    $record['recorded_at_utc'] = gmdate('c');
    $record['actor_admin_id'] = $_SESSION['admin_id'] ?? $_SESSION['user_id'] ?? null;
    $record['actor_email'] = $_SESSION['admin_email'] ?? $_SESSION['email'] ?? null;
    $record['ip'] = $_SERVER['REMOTE_ADDR'] ?? null;
    @file_put_contents(
        $backupDir . '/pricing-manager-audit.jsonl',
        json_encode($record, JSON_UNESCAPED_SLASHES) . "\n",
        FILE_APPEND | LOCK_EX
    );
}

$db = klevr_pricing_db();
$catalogPath = klevr_pricing_catalog_path($root);
$backupDir = $root . '/storage/pricing-backups';
$notice = null;
$error = null;
$saveSummary = [];

try {
    $catalog = klevr_pricing_load_catalog($catalogPath);
} catch (Throwable $exception) {
    $catalog = [];
    $error = $exception->getMessage();
}
$catalogEntries = klevr_pricing_collect_catalog($catalog);
$catalogByToken = [];
foreach ($catalogEntries as $entry) {
    $catalogByToken[klevr_pricing_token($entry['path'])] = $entry;
}

$tableData = [];
if ($db instanceof PDO) {
    foreach (['packages', 'monitoring_plans', 'products', 'hardware_addons'] as $table) {
        $tableData[$table] = klevr_pricing_load_table($db, $table);
    }
}

if ($_SERVER['REQUEST_METHOD'] === 'POST' && isset($_POST['save_pricing'])) {
    try {
        if (!klevr_pricing_verify_csrf((string)($_POST['csrf_token'] ?? ''))) {
            throw new RuntimeException('Your session expired. Refresh the page and try again.');
        }
        if (!$catalog) {
            throw new RuntimeException('The pricing catalog is unavailable.');
        }

        $catalogChanges = [];
        $newCatalog = $catalog;
        foreach ((array)($_POST['catalog_price'] ?? []) as $token => $rawValue) {
            if (!isset($catalogByToken[$token])) {
                continue;
            }
            $rawValue = trim((string)$rawValue);
            if (!is_numeric($rawValue)) {
                throw new InvalidArgumentException('Every equipment price must be a valid number.');
            }
            $newPrice = round((float)$rawValue, 2);
            if ($newPrice < 0 || $newPrice > 1000000) {
                throw new InvalidArgumentException('Prices must be between $0.00 and $1,000,000.00.');
            }
            $entry = $catalogByToken[$token];
            $oldPrice = round((float)$entry['customer_price'], 2);
            if ($oldPrice === $newPrice) {
                continue;
            }
            $decodedPath = klevr_pricing_path_from_token((string)$token);
            if ($decodedPath === null || !klevr_pricing_set_catalog_price($newCatalog, $decodedPath, $newPrice)) {
                throw new RuntimeException('An equipment record could not be updated safely.');
            }
            $catalogChanges[] = [
                'source' => 'catalog',
                'token' => $token,
                'name' => $entry['name'],
                'sku' => $entry['sku'],
                'old' => $oldPrice,
                'new' => $newPrice,
                'entry' => $entry,
            ];
        }

        if (!$catalogChanges && empty($_POST['db_price'])) {
            throw new RuntimeException('No pricing changes were submitted.');
        }

        $catalogBackup = null;
        if ($catalogChanges) {
            $catalogBackup = klevr_pricing_write_catalog($catalogPath, $newCatalog, $backupDir);
        }

        $databaseChanges = [];
        try {
            if ($db instanceof PDO) {
                $db->beginTransaction();
                $databaseChanges = klevr_pricing_update_database($db, (array)($_POST['db_price'] ?? []), $tableData);
                foreach ($catalogChanges as &$catalogChange) {
                    $catalogChange['synced_to'] = klevr_pricing_sync_catalog_item($db, $catalogChange['entry'], (float)$catalogChange['new']);
                    unset($catalogChange['entry']);
                }
                unset($catalogChange);
                $db->commit();
            }
        } catch (Throwable $databaseError) {
            if ($db instanceof PDO && $db->inTransaction()) {
                $db->rollBack();
            }
            if ($catalogBackup && is_file($catalogBackup)) {
                @copy($catalogBackup, $catalogPath);
            }
            throw $databaseError;
        }

        $allChanges = array_merge($catalogChanges, $databaseChanges);
        if (!$allChanges) {
            throw new RuntimeException('No pricing values changed.');
        }
        klevr_pricing_append_audit($backupDir, [
            'event' => 'pricing_saved',
            'catalog_backup' => $catalogBackup,
            'changes' => $allChanges,
        ]);
        $saveSummary = $allChanges;
        $notice = count($allChanges) . ' pricing change' . (count($allChanges) === 1 ? '' : 's') . ' saved. New customer quotes, carts, and checkouts now use the updated values.';

        $catalog = klevr_pricing_load_catalog($catalogPath);
        $catalogEntries = klevr_pricing_collect_catalog($catalog);
        $catalogByToken = [];
        foreach ($catalogEntries as $entry) {
            $catalogByToken[klevr_pricing_token($entry['path'])] = $entry;
        }
        if ($db instanceof PDO) {
            foreach (array_keys($tableData) as $table) {
                $tableData[$table] = klevr_pricing_load_table($db, $table);
            }
        }
    } catch (Throwable $exception) {
        $error = $exception->getMessage();
        klevr_pricing_append_audit($backupDir, [
            'event' => 'pricing_save_failed',
            'message' => $exception->getMessage(),
        ]);
    }
}

$query = trim((string)($_GET['q'] ?? ''));
$matchesQuery = static function (string $text) use ($query): bool {
    return $query === '' || stripos($text, $query) !== false;
};

$packageRows = [];
$equipmentRows = [];
foreach ($tableData as $table => $data) {
    foreach ($data['rows'] as $row) {
        $label = klevr_pricing_row_name($row, $table);
        if (!$matchesQuery($label . ' ' . implode(' ', array_map('strval', $row)))) {
            continue;
        }
        $record = ['table' => $table, 'row' => $row, 'price_columns' => $data['price_columns']];
        if (klevr_pricing_is_package_row($table, $row)) {
            $packageRows[] = $record;
        } else {
            $equipmentRows[] = $record;
        }
    }
}
$filteredCatalogEntries = array_values(array_filter($catalogEntries, static function (array $entry) use ($matchesQuery): bool {
    return $matchesQuery($entry['name'] . ' ' . $entry['sku'] . ' ' . implode(' / ', array_map('strval', $entry['path'])));
}));

if ($headerHtml !== '') {
    echo $headerHtml;
} else {
    echo '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pricing Manager</title></head><body>';
}
?>
<style>
.kpm-wrap{max-width:1500px;margin:0 auto;padding:28px}.kpm-hero{background:linear-gradient(135deg,#370877,#5d18a7);color:#fff;border-radius:22px;padding:28px;box-shadow:0 18px 50px rgba(55,8,119,.22)}.kpm-hero h1{margin:0 0 8px;font-size:clamp(1.8rem,3vw,2.7rem)}.kpm-hero p{max-width:900px;margin:0;color:rgba(255,255,255,.84);line-height:1.6}.kpm-pills{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}.kpm-pill{padding:8px 12px;border-radius:999px;background:rgba(255,255,255,.13);font-size:.86rem}.kpm-alert{margin:18px 0;padding:14px 16px;border-radius:12px;border:1px solid}.kpm-alert.ok{background:#f1fbe8;border-color:#a8d77a;color:#24510b}.kpm-alert.error{background:#fff1f3;border-color:#ef9ba9;color:#8b1730}.kpm-toolbar{display:flex;gap:12px;align-items:end;justify-content:space-between;flex-wrap:wrap;margin:24px 0}.kpm-toolbar form{display:flex;gap:8px;flex-wrap:wrap;align-items:end}.kpm-field label{display:block;font-weight:700;font-size:.82rem;margin-bottom:6px}.kpm-field input[type=search]{min-width:280px;padding:11px 13px;border:1px solid #d8d2df;border-radius:10px}.kpm-btn{border:0;border-radius:10px;padding:11px 16px;font-weight:800;cursor:pointer;text-decoration:none;display:inline-flex;align-items:center;justify-content:center}.kpm-btn.primary{background:#370877;color:#fff}.kpm-btn.secondary{background:#eef8e5;color:#315d12}.kpm-section{background:#fff;border:1px solid #e5e0e9;border-radius:18px;margin:20px 0;box-shadow:0 10px 30px rgba(40,24,58,.06);overflow:hidden}.kpm-section-head{padding:20px 22px;border-bottom:1px solid #ece8ef;display:flex;justify-content:space-between;gap:18px;align-items:center}.kpm-section-head h2{margin:0;color:#271832}.kpm-section-head p{margin:4px 0 0;color:#756c7b}.kpm-count{white-space:nowrap;background:#eff8e7;color:#315d12;border-radius:999px;padding:7px 11px;font-weight:800;font-size:.82rem}.kpm-table-wrap{overflow:auto}.kpm-table{width:100%;border-collapse:collapse;min-width:860px}.kpm-table th,.kpm-table td{padding:13px 15px;border-bottom:1px solid #efebf1;text-align:left;vertical-align:middle}.kpm-table th{font-size:.76rem;text-transform:uppercase;letter-spacing:.05em;color:#6e6474;background:#faf9fb}.kpm-name{font-weight:800;color:#271832}.kpm-meta{font-size:.78rem;color:#807686;margin-top:3px}.kpm-money{width:120px;padding:9px 10px;border:1px solid #d9d2df;border-radius:9px;font-variant-numeric:tabular-nums}.kpm-money:focus{outline:3px solid rgba(118,189,34,.24);border-color:#76bd22}.kpm-margin{font-weight:800}.kpm-margin.low{color:#ad2945}.kpm-margin.good{color:#33730e}.kpm-empty{padding:28px;color:#756c7b}.kpm-savebar{position:sticky;bottom:14px;z-index:20;margin:24px 0 0;background:#21122d;color:#fff;border-radius:16px;padding:14px 16px;display:flex;align-items:center;justify-content:space-between;gap:16px;box-shadow:0 18px 45px rgba(33,18,45,.3)}.kpm-savebar small{color:#d8cfe0}.kpm-savebar .kpm-btn{background:#76bd22;color:#152408;min-width:180px}.kpm-changed{display:none;color:#dfffbd;font-weight:800;margin-left:10px}.kpm-summary{margin:12px 0 0;padding-left:20px}.kpm-summary li{margin:5px 0}.kpm-help{background:#faf8fc;border:1px solid #e4ddea;border-radius:14px;padding:16px;line-height:1.55;color:#62576a}.kpm-source{display:inline-block;padding:4px 7px;border-radius:6px;background:#f1edf5;color:#5a3c6f;font-size:.72rem;font-weight:800;text-transform:uppercase;letter-spacing:.04em}@media(max-width:700px){.kpm-wrap{padding:16px}.kpm-hero{padding:22px}.kpm-field input[type=search]{min-width:0;width:100%}.kpm-savebar{align-items:stretch;flex-direction:column}.kpm-savebar .kpm-btn{width:100%}}
</style>
<main class="kpm-wrap">
    <section class="kpm-hero">
        <h1>KLEVR Pricing Manager</h1>
        <p>Adjust package, monitoring, installation, and equipment prices from one screen. Catalog-controlled AJAX prices update the shared source used by Packages, System Builder, cart, checkout, and new order calculations.</p>
        <div class="kpm-pills">
            <span class="kpm-pill">Protected catalog backups</span>
            <span class="kpm-pill">Prepared database updates</span>
            <span class="kpm-pill">Price-change audit log</span>
            <span class="kpm-pill">Historical orders unchanged</span>
        </div>
    </section>

    <?php if ($notice): ?>
        <div class="kpm-alert ok"><?= klevr_pricing_h($notice) ?></div>
        <?php if ($saveSummary): ?>
            <ul class="kpm-summary">
                <?php foreach (array_slice($saveSummary, 0, 12) as $change): ?>
                    <li><?= klevr_pricing_h($change['name'] ?? (($change['table'] ?? '') . ' #' . ($change['id'] ?? ''))) ?>: <?= klevr_pricing_h($change['field'] ?? 'customer price') ?> changed from <?= $change['old'] === null ? 'blank' : '$' . number_format((float)$change['old'], 2) ?> to <?= $change['new'] === null ? 'blank' : '$' . number_format((float)$change['new'], 2) ?>.</li>
                <?php endforeach; ?>
            </ul>
        <?php endif; ?>
    <?php endif; ?>
    <?php if ($error): ?><div class="kpm-alert error"><?= klevr_pricing_h($error) ?></div><?php endif; ?>
    <?php if (!$db): ?><div class="kpm-alert error">The database connection could not be detected. Catalog equipment prices remain visible, but database-backed package and add-on prices cannot be saved until the connection is available.</div><?php endif; ?>

    <div class="kpm-toolbar">
        <form method="get">
            <div class="kpm-field"><label for="pricing-search">Find a package or device</label><input id="pricing-search" type="search" name="q" value="<?= klevr_pricing_h($query) ?>" placeholder="Search name, SKU, category..."></div>
            <button class="kpm-btn primary" type="submit">Search</button>
            <?php if ($query !== ''): ?><a class="kpm-btn secondary" href="pricing_manager.php">Clear</a><?php endif; ?>
        </form>
        <div class="kpm-help"><strong>Live behavior:</strong> saved values apply to new customer quotes and transactions. Existing completed orders, payments, subscriptions, and monitoring accounts are not repriced.</div>
    </div>

    <form method="post" id="kpm-pricing-form">
        <input type="hidden" name="csrf_token" value="<?= klevr_pricing_h(klevr_pricing_csrf_token()) ?>">
        <input type="hidden" name="save_pricing" value="1">

        <section class="kpm-section">
            <div class="kpm-section-head"><div><h2>Package & Service Pricing</h2><p>Customer-facing packages, monitoring rates, installation, activation, and service fees stored in the database.</p></div><span class="kpm-count"><?= count($packageRows) ?> records</span></div>
            <?php if ($packageRows): ?>
            <div class="kpm-table-wrap"><table class="kpm-table"><thead><tr><th>Package / plan</th><th>Source</th><th>Price fields</th></tr></thead><tbody>
            <?php foreach ($packageRows as $record): $table = $record['table']; $row = $record['row']; ?>
                <tr><td><div class="kpm-name"><?= klevr_pricing_h(klevr_pricing_row_name($row, $table)) ?></div><div class="kpm-meta"><?= klevr_pricing_h(($row['sku'] ?? $row['slug'] ?? '') . (isset($row['id']) ? ' · ID ' . $row['id'] : '')) ?></div></td><td><span class="kpm-source"><?= klevr_pricing_h($table) ?></span></td><td>
                <?php foreach ($record['price_columns'] as $column): $value = $row[$column] ?? null; ?>
                    <label class="kpm-field" style="display:inline-block;margin:4px 12px 4px 0"><span><?= klevr_pricing_h(ucwords(str_replace('_', ' ', $column))) ?></span><input class="kpm-money" type="number" min="0" max="1000000" step="0.01" name="db_price[<?= klevr_pricing_h($table) ?>][<?= klevr_pricing_h($row['id']) ?>][<?= klevr_pricing_h($column) ?>]" value="<?= $value === null ? '' : klevr_pricing_h(number_format((float)$value, 2, '.', '')) ?>"></label>
                <?php endforeach; ?></td></tr>
            <?php endforeach; ?>
            </tbody></table></div>
            <?php else: ?><div class="kpm-empty">No package or service price records matched. The manager only exposes recognized numeric price columns and never guesses at unrelated fields.</div><?php endif; ?>
        </section>

        <section class="kpm-section">
            <div class="kpm-section-head"><div><h2>AJAX Equipment Catalog</h2><p>The authoritative customer price for catalog-controlled devices. Package equipment totals recalculate from this shared catalog.</p></div><span class="kpm-count"><?= count($filteredCatalogEntries) ?> devices</span></div>
            <?php if ($filteredCatalogEntries): ?>
            <div class="kpm-table-wrap"><table class="kpm-table"><thead><tr><th>Equipment</th><th>Internal cost</th><th>Customer price</th><th>Gross margin</th><th>Install reference</th></tr></thead><tbody>
            <?php foreach ($filteredCatalogEntries as $entry): $token = klevr_pricing_token($entry['path']); $margin = $entry['cost'] !== null && $entry['customer_price'] > 0 ? (($entry['customer_price'] - $entry['cost']) / $entry['customer_price']) * 100 : null; ?>
                <tr><td><div class="kpm-name"><?= klevr_pricing_h($entry['name']) ?></div><div class="kpm-meta"><?= klevr_pricing_h($entry['sku'] !== '' ? $entry['sku'] : implode(' / ', array_map('strval', $entry['path']))) ?></div></td><td><?= $entry['cost'] === null ? '—' : '$' . number_format($entry['cost'], 2) ?></td><td><input class="kpm-money kpm-catalog-price" data-cost="<?= $entry['cost'] === null ? '' : klevr_pricing_h((string)$entry['cost']) ?>" type="number" min="0" max="1000000" step="0.01" name="catalog_price[<?= klevr_pricing_h($token) ?>]" value="<?= klevr_pricing_h(number_format($entry['customer_price'], 2, '.', '')) ?>"></td><td><span class="kpm-margin <?= $margin !== null && $margin < 10 ? 'low' : 'good' ?>"><?= $margin === null ? '—' : number_format($margin, 2) . '%' ?></span></td><td><?= $entry['installation_price'] === null ? '—' : '$' . number_format($entry['installation_price'], 2) ?></td></tr>
            <?php endforeach; ?>
            </tbody></table></div>
            <?php else: ?><div class="kpm-empty">No catalog equipment matched this search.</div><?php endif; ?>
        </section>

        <section class="kpm-section">
            <div class="kpm-section-head"><div><h2>Other Storefront Equipment</h2><p>Database-backed cameras, add-ons, and products not controlled by the AJAX catalog.</p></div><span class="kpm-count"><?= count($equipmentRows) ?> records</span></div>
            <?php if ($equipmentRows): ?>
            <div class="kpm-table-wrap"><table class="kpm-table"><thead><tr><th>Equipment</th><th>Source</th><th>Price fields</th></tr></thead><tbody>
            <?php foreach ($equipmentRows as $record): $table = $record['table']; $row = $record['row']; ?>
                <tr><td><div class="kpm-name"><?= klevr_pricing_h(klevr_pricing_row_name($row, $table)) ?></div><div class="kpm-meta"><?= klevr_pricing_h(($row['sku'] ?? $row['slug'] ?? '') . (isset($row['id']) ? ' · ID ' . $row['id'] : '')) ?></div></td><td><span class="kpm-source"><?= klevr_pricing_h($table) ?></span></td><td>
                <?php foreach ($record['price_columns'] as $column): $value = $row[$column] ?? null; ?>
                    <label class="kpm-field" style="display:inline-block;margin:4px 12px 4px 0"><span><?= klevr_pricing_h(ucwords(str_replace('_', ' ', $column))) ?></span><input class="kpm-money" type="number" min="0" max="1000000" step="0.01" name="db_price[<?= klevr_pricing_h($table) ?>][<?= klevr_pricing_h($row['id']) ?>][<?= klevr_pricing_h($column) ?>]" value="<?= $value === null ? '' : klevr_pricing_h(number_format((float)$value, 2, '.', '')) ?>"></label>
                <?php endforeach; ?></td></tr>
            <?php endforeach; ?>
            </tbody></table></div>
            <?php else: ?><div class="kpm-empty">No additional storefront equipment matched.</div><?php endif; ?>
        </section>

        <div class="kpm-savebar"><div><strong>Review before saving.</strong><span class="kpm-changed" id="kpm-changed">Unsaved changes</span><br><small>A catalog backup is created before every equipment-price save.</small></div><button class="kpm-btn" type="submit">Save Pricing Changes</button></div>
    </form>
</main>
<script>
(function(){
  var form=document.getElementById('kpm-pricing-form');
  if(!form)return;
  var changed=document.getElementById('kpm-changed');
  form.addEventListener('input',function(event){
    changed.style.display='inline';
    var input=event.target;
    if(input.classList.contains('kpm-catalog-price')){
      var cost=parseFloat(input.getAttribute('data-cost'));
      var price=parseFloat(input.value);
      var cell=input.closest('tr').querySelector('.kpm-margin');
      if(cell&&!isNaN(cost)&&!isNaN(price)&&price>0){
        var margin=((price-cost)/price)*100;
        cell.textContent=margin.toFixed(2)+'%';
        cell.classList.toggle('low',margin<10);
        cell.classList.toggle('good',margin>=10);
      }
    }
  });
  form.addEventListener('submit',function(event){
    if(!window.confirm('Save these prices to the customer-facing website?'))event.preventDefault();
  });
})();
</script>
<?php
$footerPath = __DIR__ . '/includes/footer.php';
if (is_file($footerPath)) {
    require $footerPath;
} elseif ($headerHtml === '') {
    echo '</body></html>';
}
