<?php
/**
 * DEFENDR OS 3.5.0 — Portal/Admin UI Update Installer
 *
 * Upload this file and the included assets/ directory to the DEFENDR OS root,
 * visit this file once in a browser, then remove it after a successful update.
 */
declare(strict_types=1);

$root = __DIR__;
$version = '3.5.0';
$marker = 'DEFENDR-UI-' . $version;
$results = [];
$errors = [];

function ui_result(string $message, string $status = 'ok'): void
{
    global $results;
    $results[] = ['message' => $message, 'status' => $status];
}

function ui_error(string $message): void
{
    global $errors;
    $errors[] = $message;
}

function ui_backup(string $file, string $backupRoot, string $root): bool
{
    if (!is_file($file)) {
        return false;
    }
    $relative = ltrim(str_replace($root, '', $file), DIRECTORY_SEPARATOR);
    $target = $backupRoot . DIRECTORY_SEPARATOR . $relative;
    $directory = dirname($target);
    if (!is_dir($directory) && !mkdir($directory, 0775, true) && !is_dir($directory)) {
        return false;
    }
    return copy($file, $target);
}

function ui_inject_assets(string $file, string $marker, string $backupRoot, string $root): bool
{
    $contents = @file_get_contents($file);
    if ($contents === false) {
        ui_error('Could not read ' . $file);
        return false;
    }
    if (strpos($contents, $marker) !== false) {
        ui_result('Already connected: ' . str_replace($root . DIRECTORY_SEPARATOR, '', $file), 'skip');
        return true;
    }

    $snippet = "\n    <!-- {$marker} -->\n"
        . "    <link rel=\"stylesheet\" href=\"../assets/css/control-center-3.5.0.css?v=350\">\n"
        . "    <script defer src=\"../assets/js/control-center-3.5.0.js?v=350\"></script>\n";

    if (stripos($contents, '</head>') !== false) {
        $updated = preg_replace('/<\/head>/i', $snippet . '</head>', $contents, 1);
    } else {
        $updated = $contents . $snippet;
    }

    if (!is_string($updated)) {
        ui_error('Could not prepare update for ' . $file);
        return false;
    }

    if (!ui_backup($file, $backupRoot, $root)) {
        ui_error('Could not back up ' . $file);
        return false;
    }

    if (@file_put_contents($file, $updated, LOCK_EX) === false) {
        ui_error('Could not write ' . $file);
        return false;
    }

    ui_result('Connected clean UI assets to ' . str_replace($root . DIRECTORY_SEPARATOR, '', $file));
    return true;
}

function ui_append_loader(string $file, string $marker, string $backupRoot, string $root): bool
{
    if (!is_file($file)) {
        return false;
    }
    $contents = @file_get_contents($file);
    if ($contents === false) {
        return false;
    }
    if (strpos($contents, $marker) !== false) {
        return true;
    }

    $loader = "\n/* {$marker} fallback loader */\n"
        . "(function(){if(!/\\/(admin|portal)\\//i.test(location.pathname))return;"
        . "var b=(document.currentScript&&document.currentScript.src||'').replace(/assets\\/js\\/app\\.js.*$/,'');"
        . "if(!document.querySelector('link[data-defendr-ui]')){var l=document.createElement('link');l.rel='stylesheet';l.href=b+'assets/css/control-center-3.5.0.css?v=350';l.dataset.defendrUi='3.5.0';document.head.appendChild(l);}"
        . "if(!document.querySelector('script[data-defendr-ui]')){var s=document.createElement('script');s.src=b+'assets/js/control-center-3.5.0.js?v=350';s.defer=true;s.dataset.defendrUi='3.5.0';document.head.appendChild(s);}})();\n";

    if (!ui_backup($file, $backupRoot, $root)) {
        return false;
    }
    if (@file_put_contents($file, $contents . $loader, LOCK_EX) === false) {
        return false;
    }
    ui_result('Added fallback loader to assets/js/app.js');
    return true;
}

if (PHP_SAPI !== 'cli') {
    header('Content-Type: text/html; charset=UTF-8');
}

$requiredAssets = [
    $root . '/assets/css/control-center-3.5.0.css',
    $root . '/assets/js/control-center-3.5.0.js',
];
foreach ($requiredAssets as $asset) {
    if (!is_file($asset)) {
        ui_error('Missing required update asset: ' . str_replace($root . '/', '', $asset));
    }
}

$timestamp = date('Ymd-His');
$backupRoot = $root . '/storage/backups/ui-3.5.0-' . $timestamp;
if (!is_dir($backupRoot) && !mkdir($backupRoot, 0775, true) && !is_dir($backupRoot)) {
    ui_error('Could not create backup directory: ' . $backupRoot);
}

$headerCandidates = [
    $root . '/portal/_header.php',
    $root . '/portal/header.php',
    $root . '/admin/_header.php',
    $root . '/admin/header.php',
];

$updatedHeaders = 0;
if (!$errors) {
    foreach ($headerCandidates as $candidate) {
        if (is_file($candidate) && ui_inject_assets($candidate, $marker, $backupRoot, $root)) {
            $updatedHeaders++;
        }
    }

    if ($updatedHeaders === 0) {
        $fallback = $root . '/assets/js/app.js';
        if (!ui_append_loader($fallback, $marker, $backupRoot, $root)) {
            ui_error('No shared portal/admin header was found and the app.js fallback could not be updated.');
        }
    } else {
        ui_append_loader($root . '/assets/js/app.js', $marker, $backupRoot, $root);
    }
}

$versionFile = $root . '/VERSION';
if (!$errors && is_file($versionFile)) {
    ui_backup($versionFile, $backupRoot, $root);
    @file_put_contents($versionFile, $version . PHP_EOL, LOCK_EX);
    ui_result('Updated VERSION to ' . $version);
}

$changeLog = $root . '/CHANGELOG.md';
if (!$errors && is_file($changeLog)) {
    $existing = @file_get_contents($changeLog) ?: '';
    if (strpos($existing, '## 3.5.0') === false) {
        ui_backup($changeLog, $backupRoot, $root);
        $entry = "## 3.5.0 — Unified Portal & Admin Design\n\n"
            . "- Standardized portal and owner-admin spacing, cards, forms, tables and buttons.\n"
            . "- Added balanced responsive grids and equal-height dashboard panels.\n"
            . "- Added clean mobile navigation behavior and table overflow handling.\n"
            . "- Preserved all existing application routes and business logic.\n\n";
        @file_put_contents($changeLog, $entry . $existing, LOCK_EX);
        ui_result('Updated CHANGELOG.md');
    }
}

$success = count($errors) === 0;
?><!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DEFENDR OS UI Update</title>
<style>
:root{--navy:#061426;--blue:#155eef;--aqua:#12a8a0;--bg:#f4f7fb;--border:#dbe4ee;--text:#102a43;--muted:#627d98}*{box-sizing:border-box}body{margin:0;background:var(--bg);font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text)}main{width:min(780px,calc(100% - 32px));margin:54px auto}.shell{background:#fff;border:1px solid var(--border);border-radius:18px;box-shadow:0 18px 55px rgba(6,20,38,.11);overflow:hidden}.hero{padding:28px 30px;background:linear-gradient(135deg,var(--navy),#103756);color:#fff}.hero h1{margin:0 0 8px;font-size:28px}.hero p{margin:0;color:#c8d9e8}.body{padding:26px 30px}.status{display:inline-flex;padding:6px 10px;border-radius:999px;font-size:12px;font-weight:800;background:<?= $success ? '#dff8f4' : '#ffebeb' ?>;color:<?= $success ? '#16805f' : '#a52f2f' ?>}.list{display:grid;gap:9px;margin-top:18px}.item{padding:12px 14px;border:1px solid var(--border);border-radius:10px;background:#f9fbfd}.item.ok:before{content:'✓';color:#16805f;font-weight:900;margin-right:9px}.item.skip:before{content:'•';color:var(--muted);font-weight:900;margin-right:9px}.error{padding:12px 14px;border:1px solid #f2c1c1;border-radius:10px;background:#fff0f0;color:#982d2d}.note{margin-top:20px;padding:15px;border-left:4px solid var(--blue);background:#eef4ff;color:#244e90;border-radius:0 9px 9px 0}code{background:#edf2f7;padding:2px 5px;border-radius:5px}a{color:var(--blue)}</style>
</head>
<body>
<main>
  <section class="shell">
    <div class="hero"><h1>DEFENDR OS 3.5.0</h1><p>Unified Portal & Owner Admin Design System</p></div>
    <div class="body">
      <span class="status"><?= $success ? 'Update completed' : 'Update needs attention' ?></span>
      <?php if ($errors): ?>
        <div class="list">
          <?php foreach ($errors as $error): ?><div class="error"><?= htmlspecialchars($error, ENT_QUOTES, 'UTF-8') ?></div><?php endforeach; ?>
        </div>
      <?php endif; ?>
      <?php if ($results): ?>
        <div class="list">
          <?php foreach ($results as $result): ?><div class="item <?= htmlspecialchars($result['status'], ENT_QUOTES, 'UTF-8') ?>"><?= htmlspecialchars($result['message'], ENT_QUOTES, 'UTF-8') ?></div><?php endforeach; ?>
        </div>
      <?php endif; ?>
      <div class="note">
        Backup created at <code><?= htmlspecialchars(str_replace($root . '/', '', $backupRoot), ENT_QUOTES, 'UTF-8') ?></code>.<br>
        After confirming the portal and admin, remove <code>apply-ui-3.5.0.php</code> from the server.
      </div>
    </div>
  </section>
</main>
</body>
</html>
