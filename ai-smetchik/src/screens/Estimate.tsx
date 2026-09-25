import React, { useState } from 'react';
import { Alert, Image, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Button, Card, Chip, Header, Screen } from '../components/ui';
import { colors, radius } from '../theme';
import { formatMoney, imgSource, useStore } from '../store';
import type { ScreenProps } from '../navigation/types';

export function Estimate({ navigation }: ScreenProps<'Estimate'>) {
  const { draft, works, materials, updateWorkQty, worksTotal, materialsTotal, total, approveDraft } = useStore();
  const [editing, setEditing] = useState(false);
  const [draftNo] = useState(() => 1042);

  const shortTitle = draft.description.trim().split(/[.,\n]/)[0].slice(0, 48) || 'Скол керамической плитки';

  const onApprove = () => {
    approveDraft(shortTitle, total);
    Alert.alert('Смета согласована', `Черновик №${draftNo} отправлен в работу. Заявка появилась на дашборде.`);
    navigation.popToTop();
  };

  const onExport = () => {
    Alert.alert('Экспорт', 'В пилоте: выгрузка ВОР в 1С:Смета и PDF. В прототипе — заглушка.');
  };

  return (
    <Screen>
      <Header title={`Черновик сметы №${draftNo}`} onBack={() => navigation.goBack()} />
      <ScrollView contentContainerStyle={styles.scroll} keyboardShouldPersistTaps="handled">
        <View style={styles.accuracyBar}>
          <Ionicons name="checkmark-circle" size={17} color="#FFF" />
          <Text style={styles.accuracyText}>Точность совпадения: 92%</Text>
        </View>

        <Card style={styles.defectCard}>
          {draft.photoUri ? (
            <Image source={imgSource(draft.photoUri) as never} style={styles.defectPhoto} resizeMode="cover" />
          ) : (
            <Image source={require('../../assets/tile-chip.png')} style={styles.defectPhoto} resizeMode="cover" />
          )}
          <View style={{ flex: 1 }}>
            <Text style={styles.defectTitle} numberOfLines={2}>{shortTitle}</Text>
            <Text style={styles.defectDesc} numberOfLines={2}>
              {draft.description || 'Скол керамической плитки на полу в коридоре'}
            </Text>
            <View style={{ marginTop: 8 }}>
              <Chip label={draft.house} icon="location-outline" />
            </View>
          </View>
        </Card>

        <Text style={styles.section}>
          {editing ? 'Работы (ВОР) — объёмы' : 'Работы (ВОР)'}
        </Text>
        <Card style={{ padding: 0, overflow: 'hidden' }}>
          <View style={styles.tableHead}>
            <Text style={[styles.th, { flex: 2 }]}>Наименование работ</Text>
            <Text style={[styles.th, { flex: 0.9, textAlign: 'right' }]}>Объём</Text>
            <Text style={[styles.th, { flex: 0.9, textAlign: 'right' }]}>Цена</Text>
            <Text style={[styles.th, { flex: 0.9, textAlign: 'right' }]}>Сумма</Text>
          </View>
          {works.map((w, i) => (
            <View key={w.id} style={[styles.tableRow, i > 0 && styles.rowBorder]}>
              <View style={{ flex: 2 }}>
                <Text style={styles.wName}>{w.name}</Text>
                <Text style={styles.wRate}>{w.rate}</Text>
              </View>
              <View style={{ flex: 0.9, alignItems: 'flex-end' }}>
                {editing ? (
                  <TextInput
                    style={styles.qtyInput}
                    keyboardType="decimal-pad"
                    value={String(w.qty)}
                    onChangeText={(t) => updateWorkQty(w.id, parseFloat(t.replace(',', '.')) || 0)}
                  />
                ) : (
                  <Text style={styles.cell}>{w.qty} {w.unit}</Text>
                )}
              </View>
              <Text style={[styles.cell, { flex: 0.9, textAlign: 'right' }]}>{formatMoney(w.price)} ₽</Text>
              <Text style={[styles.cellBold, { flex: 0.9, textAlign: 'right' }]}>{formatMoney(w.qty * w.price)} ₽</Text>
            </View>
          ))}
        </Card>

        <Text style={styles.section}>Материалы</Text>
        <Card style={{ padding: 0 }}>
          {materials.map((m, i) => (
            <View key={m.id} style={[styles.matRow, i > 0 && styles.rowBorder]}>
              <View style={styles.matIcon}>
                <Ionicons name="cube-outline" size={17} color={colors.primary} />
              </View>
              <Text style={styles.matName}>{m.name}</Text>
              <Text style={styles.matQty}>{m.qty} {m.unit}</Text>
              <Text style={styles.matPrice}>{formatMoney(m.qty * m.price)} ₽</Text>
            </View>
          ))}
        </Card>

        <Card style={styles.totalCard}>
          <Text style={styles.totalLabel}>Итого, предварительная оценка</Text>
          <Text style={styles.totalValue}>{formatMoney(total)} ₽</Text>
          <Text style={styles.totalNote}>
            Работы {formatMoney(worksTotal)} ₽ + Материалы {formatMoney(materialsTotal)} ₽
          </Text>
        </Card>

        <View style={styles.actions}>
          <Button
            label={editing ? 'Готово' : 'Редактировать'}
            icon={editing ? 'checkmark-done-outline' : 'create-outline'}
            variant="outline"
            onPress={() => setEditing((v) => !v)}
            style={{ flex: 1 }}
          />
          <Button label="Согласовать" icon="checkmark-circle-outline" variant="success" onPress={onApprove} style={{ flex: 1 }} />
          <Pressable style={styles.exportBtn} onPress={onExport}>
            <Ionicons name="download-outline" size={22} color={colors.primary} />
          </Pressable>
        </View>
        <Text style={styles.exportHint}>Экспорт в 1С / PDF</Text>
        <View style={{ height: 40 }} />
      </ScrollView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  scroll: { paddingHorizontal: 16, paddingTop: 2, flexGrow: 1 },
  accuracyBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    backgroundColor: colors.success,
    borderRadius: radius.input,
    paddingVertical: 11,
    marginBottom: 12,
  },
  accuracyText: { color: '#FFF', fontSize: 14.5, fontWeight: '800' },
  defectCard: { flexDirection: 'row', gap: 12, marginBottom: 6 },
  defectPhoto: { width: 84, height: 84, borderRadius: 12, backgroundColor: colors.border },
  defectTitle: { fontSize: 15, fontWeight: '800', color: colors.text },
  defectDesc: { fontSize: 12.5, color: colors.textMuted, marginTop: 3 },
  section: { fontSize: 13, fontWeight: '700', color: colors.textMuted, marginTop: 18, marginBottom: 8, textTransform: 'uppercase', letterSpacing: 0.4 },
  tableHead: { flexDirection: 'row', backgroundColor: '#EFF6FF', padding: 12, gap: 6 },
  th: { fontSize: 11.5, fontWeight: '700', color: colors.inkSoft },
  tableRow: { flexDirection: 'row', padding: 12, gap: 6, alignItems: 'center' },
  rowBorder: { borderTopWidth: StyleSheet.hairlineWidth, borderTopColor: colors.border },
  wName: { fontSize: 13.5, fontWeight: '600', color: colors.text },
  wRate: { fontSize: 11.5, color: colors.primary, marginTop: 3, fontWeight: '600' },
  cell: { fontSize: 13, color: colors.textMuted },
  cellBold: { fontSize: 13.5, fontWeight: '800', color: colors.text },
  qtyInput: {
    borderWidth: 1,
    borderColor: colors.primary,
    borderRadius: 8,
    paddingHorizontal: 6,
    width: 62,
    fontSize: 13,
    color: colors.text,
    backgroundColor: '#EFF6FF',
  },
  matRow: { flexDirection: 'row', alignItems: 'center', padding: 12, gap: 10 },
  matIcon: {
    width: 32,
    height: 32,
    borderRadius: 9,
    backgroundColor: '#EFF6FF',
    alignItems: 'center',
    justifyContent: 'center',
  },
  matName: { flex: 1, fontSize: 13.5, fontWeight: '600', color: colors.text },
  matQty: { fontSize: 12.5, color: colors.textMuted },
  matPrice: { fontSize: 13.5, fontWeight: '700', color: colors.text, width: 70, textAlign: 'right' },
  totalCard: { backgroundColor: colors.navy, marginTop: 18 },
  totalLabel: { color: '#9CA3AF', fontSize: 13 },
  totalValue: { color: '#FFF', fontSize: 36, fontWeight: '900', marginTop: 4 },
  totalNote: { color: '#6B7280', fontSize: 11.5, marginTop: 6 },
  actions: { flexDirection: 'row', gap: 10, marginTop: 18, alignItems: 'stretch' },
  exportBtn: {
    width: 50,
    borderRadius: radius.input,
    borderWidth: 1.5,
    borderColor: colors.primary,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.surface,
  },
  exportHint: { textAlign: 'center', fontSize: 11, color: colors.textMuted, marginTop: 6 },
});
